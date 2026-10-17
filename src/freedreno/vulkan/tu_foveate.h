/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_FOVEATE_H
#define TU_FOVEATE_H

#include "tu_common.h"

#define TU_GAZE_MAGIC 0x455a4147u

struct tu_gaze_shm {
   uint32_t magic;
   uint32_t seq;
   uint64_t time_ns;
   float uv[2][2];
   uint32_t valid;
   uint32_t layout;
};

struct tu_frag_area;

bool tu_foveate_enabled(void);

bool tu_foveate_pass_eligible(struct tu_cmd_buffer *cmd);

void tu_foveate_frag_area(struct tu_cmd_buffer *cmd,
                          uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                          unsigned view, struct tu_frag_area *area);

#endif
