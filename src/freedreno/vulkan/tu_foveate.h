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
   float center[2][2];
};

struct tu_frag_area;

bool tu_foveate_enabled(void);

bool tu_foveate_dynamic_wanted(const VkRenderingInfo *info);

bool tu_foveate_pass_eligible(struct tu_cmd_buffer *cmd);

bool tu_foveate_wants_gmem(struct tu_cmd_buffer *cmd);

bool tu_foveate_fb_eligible(const struct tu_render_pass *pass,
                            const struct tu_framebuffer *fb);

uint32_t tu_foveate_max_bin(void);

#define TU_RT_FOV_PACK1D  (1u << 0)
#define TU_RT_FOV_SHARE   (1u << 1)

#define TU_RT_FOV_CMD_OFFSET 112
#define TU_RT_FOV_CMD_SIZE   64

struct tu_rt_fov_block {
   uint32_t enable;
   uint32_t lvl_1d;
   float r0;
   float r1;
   float gaze[2][2];
   float sbs;
   uint32_t max_lvl;
   uint32_t pad[6];
};

uint32_t tu_foveate_rt_mode(void);

void tu_foveate_rt_fill(struct tu_device *dev, struct tu_rt_fov_block *blk);

void tu_foveate_frag_area(struct tu_cmd_buffer *cmd,
                          uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                          unsigned view, struct tu_frag_area *area);

#endif
