/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_rt_prof.h"

#include <atomic>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include "util/os_time.h"
#include "util/u_debug.h"
#include "nir.h"
#include "ir3/ir3.h"

#define TU_RT_PROF_MAX 256

struct tu_rt_prof_entry {
   const char *name;
   uint64_t peak_kb;
   uint64_t enter_kb;
   uint64_t max_delta_kb;
   int64_t ns;
   uint32_t calls;
};

static struct {
   std::atomic<bool> active;
   std::atomic<int> cur;
   pthread_t owner;
   pthread_t sampler;
   int64_t cur_start;
   uint32_t count;
   struct tu_rt_prof_entry e[TU_RT_PROF_MAX];
   uint64_t base_kb, peak_kb;
   pthread_mutex_t lock;
} prof = { .lock = PTHREAD_MUTEX_INITIALIZER };

static uint64_t
rss_kb(void)
{
   FILE *f = fopen("/proc/self/statm", "r");
   if (!f)
      return 0;
   unsigned long size = 0, res = 0;
   if (fscanf(f, "%lu %lu", &size, &res) != 2)
      res = 0;
   fclose(f);
   return (uint64_t) res * (sysconf(_SC_PAGESIZE) / 1024);
}

static int
prof_slot(const char *name)
{
   for (uint32_t i = 0; i < prof.count; i++)
      if (prof.e[i].name == name)
         return i;
   if (prof.count == TU_RT_PROF_MAX)
      return TU_RT_PROF_MAX - 1;
   prof.e[prof.count].name = name;
   return prof.count++;
}

static void
prof_hook(const char *name)
{
   if (!prof.active.load() || !pthread_equal(pthread_self(), prof.owner))
      return;
   pthread_mutex_lock(&prof.lock);
   int64_t now = os_time_get_nano();
   uint64_t kb = rss_kb();
   int c = prof.cur.load();
   if (c >= 0) {
      prof.e[c].ns += now - prof.cur_start;
      if (kb > prof.e[c].enter_kb && kb - prof.e[c].enter_kb > prof.e[c].max_delta_kb)
         prof.e[c].max_delta_kb = kb - prof.e[c].enter_kb;
   }
   int s = prof_slot(name);
   prof.e[s].calls++;
   prof.e[s].enter_kb = kb;
   if (kb > prof.e[s].peak_kb)
      prof.e[s].peak_kb = kb;
   prof.cur_start = now;
   prof.cur.store(s);
   pthread_mutex_unlock(&prof.lock);
}

static void *
prof_sampler(void *)
{
   while (prof.active.load()) {
      uint64_t kb = rss_kb();
      pthread_mutex_lock(&prof.lock);
      if (kb > prof.peak_kb)
         prof.peak_kb = kb;
      int c = prof.cur.load();
      if (c >= 0) {
         if (kb > prof.e[c].peak_kb)
            prof.e[c].peak_kb = kb;
         if (kb > prof.e[c].enter_kb && kb - prof.e[c].enter_kb > prof.e[c].max_delta_kb)
            prof.e[c].max_delta_kb = kb - prof.e[c].enter_kb;
      }
      pthread_mutex_unlock(&prof.lock);
      usleep(10000);
   }
   return NULL;
}

void
tu_rt_prof_begin(void)
{
   if (!debug_get_bool_option("TU_RT_PROF", false))
      return;
   prof.count = 0;
   prof.cur.store(-1);
   prof.owner = pthread_self();
   prof.base_kb = prof.peak_kb = rss_kb();
   prof.cur_start = os_time_get_nano();
   prof.active.store(true);
   nir_pass_hook = prof_hook;
   ir3_pass_hook = prof_hook;
   pthread_create(&prof.sampler, NULL, prof_sampler, NULL);
}

void
tu_rt_prof_mark(const char *name, const nir_shader *nir)
{
   if (!prof.active.load())
      return;
   prof_hook(name);
   if (nir) {
      unsigned instrs = 0, blocks = 0;
      nir_foreach_function_impl (impl, nir) {
         nir_foreach_block (block, impl) {
            blocks++;
            nir_foreach_instr (instr, block)
               instrs++;
         }
      }
      mesa_logi("tu-rt-prof: %s nir instrs=%u blocks=%u ssa=%u rss=%zuM",
                name, instrs, blocks,
                nir_shader_get_entrypoint((nir_shader *) nir) ?
                   nir_shader_get_entrypoint((nir_shader *) nir)->ssa_alloc : 0,
                (size_t) (rss_kb() >> 10));
   }
}

void
tu_rt_prof_end(const char *label)
{
   if (!prof.active.load())
      return;
   prof_hook("<end>");
   prof.active.store(false);
   nir_pass_hook = NULL;
   ir3_pass_hook = NULL;
   pthread_join(prof.sampler, NULL);

   uint32_t order[TU_RT_PROF_MAX];
   for (uint32_t i = 0; i < prof.count; i++)
      order[i] = i;
   for (uint32_t i = 0; i < prof.count; i++)
      for (uint32_t j = i + 1; j < prof.count; j++)
         if (prof.e[order[j]].max_delta_kb > prof.e[order[i]].max_delta_kb) {
            uint32_t t = order[i]; order[i] = order[j]; order[j] = t;
         }
   mesa_logi("tu-rt-prof: %s base=%zuM peak=%zuM (+%zuM)", label,
             (size_t) (prof.base_kb >> 10), (size_t) (prof.peak_kb >> 10),
             (size_t) ((prof.peak_kb - prof.base_kb) >> 10));
   for (uint32_t k = 0; k < MIN2(prof.count, 20u); k++) {
      const struct tu_rt_prof_entry *e = &prof.e[order[k]];
      mesa_logi("tu-rt-prof:   %-36s grow=%6zuM peak=%6zuM calls=%4u time=%8.1fms",
                e->name, (size_t) (e->max_delta_kb >> 10), (size_t) (e->peak_kb >> 10),
                e->calls, e->ns / 1e6);
   }
   uint32_t torder[TU_RT_PROF_MAX];
   for (uint32_t i = 0; i < prof.count; i++)
      torder[i] = i;
   for (uint32_t i = 0; i < prof.count; i++)
      for (uint32_t j = i + 1; j < prof.count; j++)
         if (prof.e[torder[j]].ns > prof.e[torder[i]].ns) {
            uint32_t t = torder[i]; torder[i] = torder[j]; torder[j] = t;
         }
   for (uint32_t k = 0; k < MIN2(prof.count, 12u); k++) {
      const struct tu_rt_prof_entry *e = &prof.e[torder[k]];
      mesa_logi("tu-rt-prof:   time %-36s %8.1fms calls=%u", e->name, e->ns / 1e6, e->calls);
   }
}
