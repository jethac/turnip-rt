/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "ir3_replay.h"

#include <stdio.h>

#include "nir_serialize.h"
#include "util/blob.h"
#include "util/mesa-blake3.h"

static void
put(FILE *f, const void *p, uint32_t size)
{
   fwrite(&size, sizeof(size), 1, f);
   fwrite(p, 1, size, f);
}

void
ir3_replay_dump(const char *dir, const struct ir3_compiler *compiler,
                const nir_shader *nir,
                const struct ir3_shader_options *options,
                const struct ir3_shader_key *key)
{
   struct blob b;
   blob_init(&b);
   nir_serialize(&b, nir, false);
   if (b.out_of_memory) {
      blob_finish(&b);
      return;
   }
   unsigned char hash[BLAKE3_OUT_LEN];
   _mesa_blake3_compute(b.data, b.size, hash);
   char name[512];
   snprintf(name, sizeof(name), "%s/ir3-%02x%02x%02x%02x%02x%02x.ir3d", dir,
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5]);
   FILE *f = fopen(name, "wb");
   if (f) {
      uint32_t hdr[2] = { IR3_REPLAY_MAGIC, IR3_REPLAY_VERSION };
      fwrite(hdr, sizeof(hdr), 1, f);
      put(f, compiler->dev_id, sizeof(*compiler->dev_id));
      put(f, &compiler->options, sizeof(compiler->options));
      put(f, options, sizeof(*options));
      put(f, key, sizeof(*key));
      put(f, b.data, b.size);
      fclose(f);
      mesa_logi("ir3: dumped %s (%zu bytes nir)", name, b.size);
   }
   blob_finish(&b);
}
