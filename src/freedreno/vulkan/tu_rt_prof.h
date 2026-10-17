/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_RT_PROF_H
#define TU_RT_PROF_H

#include "tu_common.h"

void tu_rt_prof_begin(void);
void tu_rt_prof_end(const char *label);
void tu_rt_prof_mark(const char *name, const nir_shader *nir);

#endif
