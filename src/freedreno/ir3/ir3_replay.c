/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "ir3_replay.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "nir_serialize.h"
#include "compiler/glsl_types.h"
#include "util/blob.h"
#include "util/os_time.h"
#include "util/mesa-blake3.h"
#include "freedreno_dev_info.h"

static void *
get(FILE *f, uint32_t expect)
{
   uint32_t size;
   if (fread(&size, sizeof(size), 1, f) != 1)
      return NULL;
   if (expect && size != expect) {
      fprintf(stderr, "size mismatch %u != %u\n", size, expect);
      exit(1);
   }
   void *p = malloc(size + 4);
   if (fread(p, 1, size, f) != size)
      return NULL;
   return p;
}

static long
maxrss_mb(void)
{
   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
   return ru.ru_maxrss / 1024;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s file.ir3d [repeat]\n", argv[0]);
      return 1;
   }
   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      perror(argv[1]);
      return 1;
   }
   uint32_t hdr[2];
   if (fread(hdr, sizeof(hdr), 1, f) != 1 || hdr[0] != IR3_REPLAY_MAGIC ||
       hdr[1] != IR3_REPLAY_VERSION) {
      fprintf(stderr, "bad header\n");
      return 1;
   }
   struct fd_dev_id *dev_id = get(f, sizeof(struct fd_dev_id));
   struct ir3_compiler_options *copts = get(f, sizeof(struct ir3_compiler_options));
   struct ir3_shader_options *sopts = get(f, sizeof(struct ir3_shader_options));
   struct ir3_shader_key *key = get(f, sizeof(struct ir3_shader_key));
   uint32_t nsize;
   if (fread(&nsize, 4, 1, f) != 1)
      return 1;
   void *ndata = malloc(nsize);
   if (fread(ndata, 1, nsize, f) != nsize)
      return 1;
   fclose(f);

   glsl_type_singleton_init_or_ref();
   copts->disable_cache = true;
   sopts->target_full_regs = 0;
   struct ir3_compiler *compiler =
      ir3_compiler_create(NULL, dev_id, fd_dev_info_raw(dev_id), copts);
   if (!compiler) {
      fprintf(stderr, "compiler create failed\n");
      return 1;
   }
   int repeat = argc > 2 ? atoi(argv[2]) : 1;
   for (int r = 0; r < repeat; r++) {
      struct blob_reader br;
      blob_reader_init(&br, ndata, nsize);
      nir_shader *nir = nir_deserialize(NULL, &compiler->nir_options, &br);
      long rss0 = maxrss_mb();
      int64_t t0 = os_time_get_nano();
      struct ir3_shader *shader = ir3_shader_from_nir(compiler, nir, sopts);
      struct ir3_shader_variant *v = ir3_shader_create_variant(shader, key, false);
      int64_t t1 = os_time_get_nano();
      unsigned char h[BLAKE3_OUT_LEN] = {0};
      if (v && v->bin)
         _mesa_blake3_compute(v->bin, v->info.size, h);
      printf("REPLAY bin=%02x%02x%02x%02x%02x%02x ", h[0], h[1], h[2], h[3], h[4], h[5]);
      printf("REPLAY %s: %s instrs=%u regs=%u half=%u stp=%u ldp=%u nops=%u mov=%u "
             "time=%.1fms maxrss=%ldM (+%ldM)\n",
             argv[1], v ? "ok" : "FAIL", v ? v->info.instrs_count : 0,
             v ? v->info.max_reg + 1 : 0, v ? v->info.max_half_reg + 1 : 0,
             v ? v->info.stp_count : 0, v ? v->info.ldp_count : 0,
             v ? v->info.nops_count : 0, v ? v->info.mov_count : 0,
             (t1 - t0) / 1e6, maxrss_mb(), maxrss_mb() - rss0);
      ir3_shader_destroy(shader);
   }
   ir3_compiler_destroy(compiler);
   return 0;
}
