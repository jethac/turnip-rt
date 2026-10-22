/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_RT_FUNNEL_H
#define TU_RT_FUNNEL_H

#include "nir.h"

struct tu_rt_funnel_info {
   unsigned sites;
   unsigned classes;
   unsigned stack_bytes;
   unsigned instrs_before;
   unsigned instrs_after;
   const char *fallback;
   char classes_desc[128];
};

struct tu_rt_funnel_options {
   unsigned max_stack_bytes;
   unsigned merge_mode;
   bool slots_in_regs;
};

nir_shader *
tu_rt_funnel_raygen(const nir_shader *raygen, void *mem_ctx,
                    const struct tu_rt_funnel_options *options,
                    struct tu_rt_funnel_info *info);

#endif
