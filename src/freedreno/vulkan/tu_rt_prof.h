/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_RT_PROF_H
#define TU_RT_PROF_H

#include "tu_common.h"
#include "util/os_time.h"

void tu_rt_prof_begin(void);
void tu_rt_prof_end(const char *label);
void tu_rt_prof_mark(const char *name, const nir_shader *nir);

enum tu_rt_phase {
   TU_RT_PH_SETUP,
   TU_RT_PH_EMIT,
   TU_RT_PH_INLINE,
   TU_RT_PH_VARS,
   TU_RT_PH_RQ,
   TU_RT_PH_OPT,
   TU_RT_PH_IO,
   TU_RT_PH_FINALIZE,
   TU_RT_PH_POST_FINALIZE,
   TU_RT_PH_LOWER_VARIANT,
   TU_RT_PH_BACKEND,
   TU_RT_PH_TAIL,
   TU_RT_PH_COUNT,
};

struct tu_rt_times {
   int64_t ns[TU_RT_PH_COUNT];
   int64_t last;
};

extern thread_local struct tu_rt_times *tu_rt_tls_times;

const char *tu_rt_phase_name(enum tu_rt_phase ph);
void tu_rt_times_install_hook(void);

static inline void
tu_rt_times_mark(enum tu_rt_phase ph)
{
   struct tu_rt_times *t = tu_rt_tls_times;
   if (!t)
      return;
   int64_t now = os_time_get_nano();
   t->ns[ph] += now - t->last;
   t->last = now;
}

#endif
