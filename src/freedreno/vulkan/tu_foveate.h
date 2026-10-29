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

/* tu-gazed v2 extension, appended after the v1 prefix above. Mirrors
 * /srv/ac8-vr-shared/foveation/gazed-v2/tu_gaze_shm.h (offsets asserted).
 * Only TU_VRS reads the v2 fields; the v1 readers are unchanged.
 */
#define TU_GAZE_SHM_VERSION 2u
#define TU_GAZE_LAYOUT_VER_MASK 0xff00u
#define TU_GAZE_LAYOUT_V2 0x0200u
#define TU_GAZE_LAYOUT_MODE_MASK 0x00ffu
#define TU_GAZE_MODE_P1_FLAT_QUAD 1u
#define TU_GAZE_MODE_P2_SBS_STEREO 2u
#define TU_GAZE_MODE_P3_HMD_PROJ 3u
#define TU_GAZE_FLAG_SACCADE (1u << 7)
#define TU_GAZE_FLAG_EYE0_NONE (1u << 11)
#define TU_GAZE_FLAG_SAMPLE_FROZEN (1u << 14)

struct tu_gaze_shm_v2 {
   struct tu_gaze_shm v1;
   uint32_t version;
   uint32_t size;
   uint32_t mode;
   uint32_t flags;
   float deg_per_uv[2][2];
   uint64_t sample_time_ns;
   uint64_t last_valid_ns;
   uint32_t sample_seq;
   uint32_t overlay_flags;
   float eye_uv_ext[2][2];
   float quad_uv[2][2];
   float gaze_dir_head[3];
   float gaze_vel_deg_s;
   float gaze_dist_m;
   uint32_t reserved[3];
};

static_assert(sizeof(struct tu_gaze_shm) == 56, "gaze shm v1 prefix");
static_assert(offsetof(struct tu_gaze_shm_v2, version) == 56, "version");
static_assert(offsetof(struct tu_gaze_shm_v2, mode) == 64, "mode");
static_assert(offsetof(struct tu_gaze_shm_v2, flags) == 68, "flags");
static_assert(offsetof(struct tu_gaze_shm_v2, deg_per_uv) == 72, "deg_per_uv");
static_assert(offsetof(struct tu_gaze_shm_v2, sample_time_ns) == 88, "sample_time_ns");
static_assert(offsetof(struct tu_gaze_shm_v2, eye_uv_ext) == 112, "eye_uv_ext");
static_assert(sizeof(struct tu_gaze_shm_v2) == 176, "gaze shm v2 size");

static inline bool
tu_gaze_shm_mode_known(const struct tu_gaze_shm_v2 *g)
{
   return (g->v1.layout & TU_GAZE_LAYOUT_VER_MASK) == TU_GAZE_LAYOUT_V2 &&
          (g->v1.layout & TU_GAZE_LAYOUT_MODE_MASK) == g->mode &&
          g->mode >= TU_GAZE_MODE_P1_FLAT_QUAD &&
          g->mode <= TU_GAZE_MODE_P3_HMD_PROJ &&
          g->size >= sizeof(struct tu_gaze_shm_v2);
}

static inline bool
tu_gaze_shm_eye_has_fovea(const struct tu_gaze_shm_v2 *g, unsigned eye)
{
   return !(g->flags & (TU_GAZE_FLAG_EYE0_NONE << eye));
}

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
   uint32_t min_w;
   uint32_t min_h;
   float sbs_max;
   float eye_min;
   float eye_max;
   uint32_t pad;
};

uint32_t tu_foveate_rt_mode(void);

#define TU_GAZE_BLOCK_DWORDS 8

struct tu_gaze_block {
   float centre[2][2];
   uint32_t valid;
   uint32_t seq;
   uint32_t pad[2];
};

bool tu_gaze_block_enabled(void);

void tu_foveate_gaze_block(struct tu_device *dev, struct tu_gaze_block *blk);

void tu_foveate_rt_fill(struct tu_device *dev, struct tu_rt_fov_block *blk);

void tu_foveate_frag_area(struct tu_cmd_buffer *cmd,
                          uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                          unsigned view, struct tu_frag_area *area);

/* TU_VRS: driver-injected attachment fragment shading rate driven by the
 * latched gaze. Mutually exclusive with TU_FOVEATE. Everything is gated on
 * tu_vrs_enabled(); with TU_VRS unset no TU_VRS code path emits anything.
 */
#define TU_VRS_TEXEL 8

enum tu_vrs_state {
   TU_VRS_OFF = 0,     /* full rate everywhere (flat map) */
   TU_VRS_ACTIVE = 1,  /* gaze map from a fresh valid sample */
   TU_VRS_HELD = 2,    /* gaze map from the last valid sample (blink hold) */
};

struct tu_vrs_frame {
   uint64_t epoch;
   int64_t latch_ns;
   int64_t sample_age_ns;   /* latch time minus gaze sample time, -1 if n/a */
   uint32_t state;          /* enum tu_vrs_state */
   uint32_t reason;         /* why OFF, see tu_vrs_reason_str() */
   bool eye[2];             /* this eye has a fovea */
   float uv[2][2];          /* per-eye gaze in eye UV, top-left origin */
   float scale[2][2];       /* degrees per eye-UV (x,y); 0 => eye-height radii */
   float r0, r1;            /* ring radii, degrees or eye-heights; r1 may be +inf */
   uint32_t cap;            /* max level: 0=1x1 1=2x2 2=4x4 */
   uint32_t dither;         /* dither band width in rate texels */
   uint32_t flags;          /* gaze flags as published (shm v2), else 0 */
};

bool tu_vrs_enabled(void);

bool tu_vrs_dims_ok(uint32_t w, uint32_t h);

bool tu_vrs_linear(void);

void tu_vrs_frame_get(struct tu_device *dev, struct tu_vrs_frame *f);

void tu_vrs_build_map(const struct tu_vrs_frame *f, uint32_t w, uint32_t h,
                      uint8_t *map, uint32_t pitch);

void tu_vrs_dump_map(const struct tu_vrs_frame *f, uint32_t w, uint32_t h,
                     const uint8_t *map, uint32_t pitch);

enum tu_vrs_pass_kind {
   TU_VRS_PASS_GAZE,      /* eligible pass, gaze map */
   TU_VRS_PASS_OFF,       /* eligible pass, flat map (gaze off/ctl off) */
   TU_VRS_PASS_FLAT,      /* ineligible pass, flat map */
   TU_VRS_PASS_APP,       /* app supplied its own attachment, untouched */
   TU_VRS_PASS_KINDS,
};

void tu_vrs_note_pass(enum tu_vrs_pass_kind kind);

struct tu_shader;
struct vk_fragment_shading_rate_state;

/* vkd3d-proton names each module with an OpString "<16 hex>.dxil" (the
 * FNV-1-64 DXIL hash it also uses for VKD3D_SHADER_OVERRIDE). Returns that
 * hash, or 0 when the SPIR-V is unavailable (module identifier) or unnamed.
 */
uint64_t tu_spirv_dxil_hash(const uint32_t *words, size_t size);
uint64_t tu_stage_dxil_hash(const VkPipelineShaderStageCreateInfo *stage);

/* TU_VRS_PS_ALLOW set (and parsed): fragment-shader allowlist mode. Passes
 * without a depth attachment can then get the gaze map (render area on
 * TU_VRS_DIMS or TU_VRS_PS_DIMS); only allowlisted draws shade coarse there.
 */
bool tu_vrs_ps_allow_mode(void);
bool tu_vrs_ps_dims_ok(uint32_t w, uint32_t h);
void tu_vrs_note_depthless_pass(void);

enum tu_vrs_pass_scope {
   TU_VRS_SCOPE_DEPTH,     /* eligible pass with depth (pre-existing rule) */
   TU_VRS_SCOPE_DEPTHLESS, /* eligible only through the PS allowlist */
   TU_VRS_SCOPE_INHERITED, /* secondary continuing a pass begun elsewhere */
};

/* Per-pipeline (bound FS + feedback-loop state) decision. Returns true and
 * rewrites *fsr when attachment FSR should be enabled for the draw.
 */
bool tu_vrs_pipeline_fsr(struct tu_shader *fs, uint32_t samples,
                         bool feedback_loop, enum tu_vrs_pass_scope scope,
                         struct vk_fragment_shading_rate_state *fsr);

#endif
