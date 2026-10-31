/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_foveate.h"

#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/os_time.h"

#include "tu_cmd_buffer.h"
#include "tu_device.h"
#include "tu_image.h"
#include "tu_pass.h"
#include "tu_shader.h"
#include "tu_vrs_ps_deny.h"
#include "vk_shader_module.h"
#include <sys/stat.h>

enum tu_gaze_src {
   TU_GAZE_SHM,
   TU_GAZE_FIXED,
   TU_GAZE_PATH,
   TU_GAZE_SWEEP,
   TU_GAZE_SACCADE,
};

#define TU_GAZE_MAX_PTS 4096
#define TU_FOVEATE_MAX_DIMS 8

struct tu_gaze_pt {
   uint32_t frame;
   float x, y;
};

struct tu_foveate_cfg {
   bool enabled;
   float r0, r1;
   float fixed_x, fixed_y;
   uint32_t min_w, min_h;
   uint32_t max_area;
   uint32_t max_bin;
   float sbs_min, sbs_max, eye_min, eye_max;
   uint32_t dims[TU_FOVEATE_MAX_DIMS][2];
   uint32_t dims_count;
   bool veto;
   uint32_t min_draws;
   uint32_t log_every;
   uint64_t stale_ns;
   const char *path;

   enum tu_gaze_src src;
   struct tu_gaze_pt *pts;
   uint32_t pts_count;
   uint32_t period;
   uint32_t hold;
   float clock_hz;
   int64_t clock_t0;
};

static struct tu_foveate_cfg cfg;
static const struct tu_gaze_shm *volatile shm;
static int64_t shm_retry_ns;

struct tu_gaze_latch {
   uint64_t key;
   float uv[2][2];
   bool valid[2];
};

static simple_mtx_t latch_mtx = SIMPLE_MTX_INITIALIZER;
static struct tu_gaze_latch latch = { UINT64_MAX, {}, {} };

static uint32_t stat_tiles[3];
static uint32_t stat_vetoed;
static uint32_t stat_last_log_frame;

static float
env_f(const char *name, float def)
{
   const char *s = os_get_option(name);
   return s ? strtof(s, NULL) : def;
}

static bool
parse_xy(const char *s, float *x, float *y)
{
   return sscanf(s, "%f,%f", x, y) == 2;
}

static void
tu_gaze_load_path(const char *file)
{
   FILE *f = fopen(file, "r");
   if (!f) {
      mesa_loge("tu-foveate: cannot open TU_GAZE path %s, using fixed gaze", file);
      cfg.src = TU_GAZE_FIXED;
      return;
   }
   cfg.pts = (struct tu_gaze_pt *) calloc(TU_GAZE_MAX_PTS, sizeof(*cfg.pts));
   char line[256];
   while (cfg.pts && cfg.pts_count < TU_GAZE_MAX_PTS && fgets(line, sizeof(line), f)) {
      struct tu_gaze_pt pt;
      if (line[0] == '#')
         continue;
      if (sscanf(line, "%u %f %f", &pt.frame, &pt.x, &pt.y) != 3)
         continue;
      if (cfg.pts_count && pt.frame < cfg.pts[cfg.pts_count - 1].frame)
         continue;
      cfg.pts[cfg.pts_count++] = pt;
   }
   fclose(f);
   if (!cfg.pts_count) {
      mesa_loge("tu-foveate: TU_GAZE path %s has no samples, using fixed gaze", file);
      cfg.src = TU_GAZE_FIXED;
      return;
   }
   cfg.period = cfg.pts[cfg.pts_count - 1].frame + 1;
}

static void
tu_gaze_parse(const char *spec)
{
   cfg.src = TU_GAZE_SHM;
   cfg.period = 240;
   cfg.hold = 60;
   if (!spec || !*spec || !strcmp(spec, "shm"))
      return;
   if (!strcmp(spec, "offcentre") || !strcmp(spec, "offcenter")) {
      cfg.src = TU_GAZE_FIXED;
      cfg.fixed_x = cfg.fixed_y = 1.0f / 3.0f;
   } else if (!strncmp(spec, "fixed:", 6)) {
      cfg.src = TU_GAZE_FIXED;
      if (!parse_xy(spec + 6, &cfg.fixed_x, &cfg.fixed_y))
         cfg.fixed_x = cfg.fixed_y = 0.5f;
   } else if (!strcmp(spec, "fixed")) {
      cfg.src = TU_GAZE_FIXED;
   } else if (!strncmp(spec, "path:", 5)) {
      cfg.src = TU_GAZE_PATH;
      tu_gaze_load_path(spec + 5);
   } else if (!strncmp(spec, "sweep", 5)) {
      cfg.src = TU_GAZE_SWEEP;
      if (spec[5] == ':')
         cfg.period = MAX2(strtoul(spec + 6, NULL, 10), 2ul);
   } else if (!strncmp(spec, "saccade:", 8)) {
      cfg.src = TU_GAZE_SACCADE;
      char *end;
      cfg.hold = MAX2(strtoul(spec + 8, &end, 10), 1ul);
      cfg.pts = (struct tu_gaze_pt *) calloc(64, sizeof(*cfg.pts));
      const char *p = *end == ':' ? end + 1 : "";
      while (cfg.pts && *p && cfg.pts_count < 64) {
         struct tu_gaze_pt pt = {};
         if (!parse_xy(p, &pt.x, &pt.y))
            break;
         cfg.pts[cfg.pts_count++] = pt;
         p = strchr(p, ';');
         if (!p)
            break;
         p++;
      }
      if (!cfg.pts_count)
         cfg.src = TU_GAZE_FIXED;
   } else {
      mesa_loge("tu-foveate: unknown TU_GAZE=%s, using fixed gaze", spec);
      cfg.src = TU_GAZE_FIXED;
   }
}

static void
tu_foveate_parse_dims(const char *s)
{
   while (s && *s && cfg.dims_count < TU_FOVEATE_MAX_DIMS) {
      uint32_t w, h;
      if (sscanf(s, "%ux%u", &w, &h) != 2)
         break;
      cfg.dims[cfg.dims_count][0] = w;
      cfg.dims[cfg.dims_count][1] = h;
      cfg.dims_count++;
      s = strchr(s, ',');
      if (s)
         s++;
   }
}

static void
tu_foveate_init_once(void)
{
   cfg.enabled = debug_get_bool_option("TU_FOVEATE", false);
   cfg.r0 = env_f("TU_FOVEATE_R0", 0.18f);
   cfg.r1 = env_f("TU_FOVEATE_R1", 0.36f);
   cfg.fixed_x = env_f("TU_FOVEATE_FIXED_X", 0.5f);
   cfg.fixed_y = env_f("TU_FOVEATE_FIXED_Y", 0.5f);
   cfg.min_w = debug_get_num_option("TU_FOVEATE_MIN_W", 1024);
   cfg.min_h = debug_get_num_option("TU_FOVEATE_MIN_H", 768);
   cfg.max_area = debug_get_num_option("TU_FOVEATE_MAX", 4);
   cfg.max_bin = debug_get_num_option("TU_FOVEATE_BIN", 256);
   cfg.sbs_min = env_f("TU_FOVEATE_SBS_MIN", 1.5f);
   cfg.sbs_max = env_f("TU_FOVEATE_SBS_MAX", 2.3f);
   cfg.eye_min = env_f("TU_FOVEATE_EYE_MIN", 0.8f);
   cfg.eye_max = env_f("TU_FOVEATE_EYE_MAX", 1.25f);
   cfg.veto = debug_get_bool_option("TU_FOVEATE_VETO", true);
   cfg.min_draws = debug_get_num_option("TU_FOVEATE_MIN_DRAWS", 0);
   cfg.log_every = debug_get_num_option("TU_FOVEATE_LOG", 0);
   tu_foveate_parse_dims(os_get_option("TU_FOVEATE_DIMS"));
   cfg.stale_ns = (uint64_t) debug_get_num_option("TU_FOVEATE_STALE_MS", 250) * 1000000ull;
   cfg.path = os_get_option("TU_FOVEATE_SHM");
   if (!cfg.path)
      cfg.path = "/dev/shm/tu-gaze";
   tu_gaze_parse(os_get_option("TU_GAZE"));
   const char *clock = os_get_option("TU_GAZE_CLOCK");
   if (clock && !strncmp(clock, "time:", 5))
      cfg.clock_hz = strtof(clock + 5, NULL);
   if (cfg.enabled)
      mesa_logi("tu-foveate: r0=%.3f r1=%.3f min=%ux%u dims=%u max=%u veto=%d min_draws=%u gaze=%d fixed=%.3f,%.3f period=%u hold=%u pts=%u clock=%.1f",
                cfg.r0, cfg.r1, cfg.min_w, cfg.min_h, cfg.dims_count, cfg.max_area,
                cfg.veto, cfg.min_draws, cfg.src, cfg.fixed_x, cfg.fixed_y, cfg.period, cfg.hold,
                cfg.pts_count, cfg.clock_hz);
}

static void
tu_foveate_init(void)
{
   static once_flag once = ONCE_FLAG_INIT;
   call_once(&once, tu_foveate_init_once);
}

bool
tu_foveate_enabled(void)
{
   tu_foveate_init();
   return cfg.enabled;
}

static const struct tu_gaze_shm *
tu_foveate_map(void)
{
   const struct tu_gaze_shm *p = shm;
   if (p)
      return p;
   int64_t now = os_time_get_nano();
   if (now < shm_retry_ns)
      return NULL;
   shm_retry_ns = now + 1000000000ll;
   int fd = open(cfg.path, O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return NULL;
   void *m = mmap(NULL, sizeof(struct tu_gaze_shm), PROT_READ, MAP_SHARED, fd, 0);
   close(fd);
   if (m == MAP_FAILED)
      return NULL;
   p = (const struct tu_gaze_shm *) m;
   if (p->magic != TU_GAZE_MAGIC) {
      munmap(m, sizeof(struct tu_gaze_shm));
      return NULL;
   }
   mesa_logi("tu-foveate: gaze source mapped");
   shm = p;
   return p;
}

static bool
tu_foveate_gaze_shm(unsigned eye, float *gx, float *gy)
{
   *gx = cfg.fixed_x;
   *gy = cfg.fixed_y;
   const struct tu_gaze_shm *p = tu_foveate_map();
   if (!p)
      return false;
   float cx = p->center[eye][0], cy = p->center[eye][1];
   if (cx > 0.f && cx < 1.f && cy > 0.f && cy < 1.f) {
      *gx = cx;
      *gy = cy;
   }
   uint32_t s0 = p->seq;
   __atomic_thread_fence(__ATOMIC_ACQUIRE);
   float x = p->uv[eye][0], y = p->uv[eye][1];
   uint32_t valid = p->valid;
   uint64_t t = p->time_ns;
   __atomic_thread_fence(__ATOMIC_ACQUIRE);
   if (s0 != p->seq || (s0 & 1) || !valid)
      return false;
   if ((uint64_t) os_time_get_nano() - t > cfg.stale_ns)
      return false;
   if (!(x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f))
      return false;
   *gx = x;
   *gy = y;
   return true;
}

static void
tu_foveate_gaze_synth(uint64_t frame, float *gx, float *gy)
{
   switch (cfg.src) {
   case TU_GAZE_PATH: {
      uint32_t f = frame % cfg.period;
      uint32_t lo = 0, hi = cfg.pts_count;
      while (hi - lo > 1) {
         uint32_t mid = (lo + hi) / 2;
         if (cfg.pts[mid].frame <= f)
            lo = mid;
         else
            hi = mid;
      }
      *gx = cfg.pts[lo].x;
      *gy = cfg.pts[lo].y;
      break;
   }
   case TU_GAZE_SWEEP: {
      float t = (float) (frame % cfg.period) / (float) cfg.period;
      float tri = t < 0.5f ? 2.f * t : 2.f - 2.f * t;
      *gx = 0.15f + 0.7f * tri;
      *gy = 0.5f;
      break;
   }
   case TU_GAZE_SACCADE: {
      const struct tu_gaze_pt *pt = &cfg.pts[(frame / cfg.hold) % cfg.pts_count];
      *gx = pt->x;
      *gy = pt->y;
      break;
   }
   default:
      *gx = cfg.fixed_x;
      *gy = cfg.fixed_y;
      break;
   }
   *gx = CLAMP(*gx, 0.f, 1.f);
   *gy = CLAMP(*gy, 0.f, 1.f);
}

static uint64_t
tu_foveate_frame(struct tu_device *dev)
{
   if (cfg.clock_hz > 0.f) {
      int64_t now = os_time_get_nano();
      int64_t t0 = p_atomic_read(&cfg.clock_t0);
      if (!t0) {
         p_atomic_cmpxchg(&cfg.clock_t0, 0, now);
         t0 = p_atomic_read(&cfg.clock_t0);
      }
      return (uint64_t) ((double) (now - t0) * 1e-9 * cfg.clock_hz);
   }
   return p_atomic_read(&dev->vk.current_frame);
}

static void
tu_foveate_log(uint64_t frame)
{
   if (!cfg.log_every || frame < (uint64_t) stat_last_log_frame + cfg.log_every)
      return;
   stat_last_log_frame = frame;
   mesa_logi("tu-foveate: frame=%" PRIu64 " gaze=%.3f,%.3f tiles 1x=%u 2x=%u 4x=%u vetoed_passes=%u",
             frame, latch.uv[0][0], latch.uv[0][1],
             p_atomic_xchg(&stat_tiles[0], 0), p_atomic_xchg(&stat_tiles[1], 0),
             p_atomic_xchg(&stat_tiles[2], 0), p_atomic_xchg(&stat_vetoed, 0));
}

static uint64_t
tu_foveate_latch(struct tu_device *dev)
{
   uint64_t frame = tu_foveate_frame(dev);
   if (latch.key != frame) {
      if (latch.key != UINT64_MAX)
         tu_foveate_log(latch.key);
      latch.key = frame;
      for (unsigned e = 0; e < 2; e++) {
         if (cfg.src == TU_GAZE_SHM) {
            latch.valid[e] =
               tu_foveate_gaze_shm(e, &latch.uv[e][0], &latch.uv[e][1]);
         } else {
            tu_foveate_gaze_synth(frame, &latch.uv[e][0], &latch.uv[e][1]);
            latch.valid[e] = true;
         }
      }
   }
   return frame;
}

static void
tu_foveate_gaze_dev(struct tu_device *dev, unsigned eye, float *gx, float *gy)
{
   simple_mtx_lock(&latch_mtx);
   tu_foveate_latch(dev);
   *gx = latch.uv[eye & 1][0];
   *gy = latch.uv[eye & 1][1];
   simple_mtx_unlock(&latch_mtx);
}

bool
tu_gaze_block_enabled(void)
{
   static const bool enabled = debug_get_bool_option("TU_GAZE_BLOCK", false);
   return enabled;
}

void
tu_foveate_gaze_block(struct tu_device *dev, struct tu_gaze_block *blk)
{
   tu_foveate_init();
   memset(blk, 0, sizeof(*blk));
   simple_mtx_lock(&latch_mtx);
   uint64_t frame = tu_foveate_latch(dev);
   for (unsigned e = 0; e < 2; e++) {
      blk->centre[e][0] = latch.uv[e][0];
      blk->centre[e][1] = latch.uv[e][1];
   }
   blk->valid = latch.valid[0] && latch.valid[1];
   blk->seq = (uint32_t) frame;
   simple_mtx_unlock(&latch_mtx);
}

static void
tu_foveate_gaze(struct tu_cmd_buffer *cmd, unsigned eye, float *gx, float *gy)
{
   tu_foveate_gaze_dev(cmd->device, eye, gx, gy);
}

struct tu_rt_fov_cfg {
   uint32_t mode;
   uint32_t lvl_1d;
   uint32_t max_lvl;
   float r0, r1;
   float sbs;
   uint32_t min_w, min_h;
};

static struct tu_rt_fov_cfg rt_cfg;

static void
tu_foveate_rt_init_once(void)
{
   tu_foveate_init();
   uint32_t mode = 0;
   if (debug_get_bool_option("TU_RT_PACK1D", false))
      mode |= TU_RT_FOV_PACK1D;
   if (debug_get_bool_option("TU_FOVEATE_RT", false))
      mode |= TU_RT_FOV_PACK1D | TU_RT_FOV_SHARE;
   rt_cfg.mode = mode;
   uint32_t run = debug_get_num_option("TU_FOVEATE_RT_1D", 1);
   rt_cfg.lvl_1d = run >= 16 ? 4 : run >= 8 ? 3 : run >= 4 ? 2 : run >= 2 ? 1 : 0;
   rt_cfg.max_lvl = MIN2(debug_get_num_option("TU_FOVEATE_RT_MAX", 4), 4);
   rt_cfg.lvl_1d = MIN2(rt_cfg.lvl_1d, rt_cfg.max_lvl);
   rt_cfg.r0 = env_f("TU_FOVEATE_RT_R0", cfg.r0);
   rt_cfg.r1 = env_f("TU_FOVEATE_RT_R1", cfg.r1);
   rt_cfg.sbs = env_f("TU_FOVEATE_RT_SBS", cfg.sbs_min);
   rt_cfg.min_w = debug_get_num_option("TU_FOVEATE_RT_MIN_W", 256);
   rt_cfg.min_h = debug_get_num_option("TU_FOVEATE_RT_MIN_H", 128);
   if (mode)
      mesa_logi("tu-foveate-rt: mode=0x%x r0=%.3f r1=%.3f max_lvl=%u lvl_1d=%u sbs=%.2f-%.2f eye=%.2f-%.2f min=%ux%u gaze=%d fixed=%.3f,%.3f",
                mode, rt_cfg.r0, rt_cfg.r1, rt_cfg.max_lvl, rt_cfg.lvl_1d, rt_cfg.sbs, cfg.sbs_max,
                cfg.eye_min, cfg.eye_max, rt_cfg.min_w, rt_cfg.min_h, cfg.src, cfg.fixed_x, cfg.fixed_y);
}

static void
tu_foveate_rt_init(void)
{
   static once_flag once = ONCE_FLAG_INIT;
   call_once(&once, tu_foveate_rt_init_once);
}

uint32_t
tu_foveate_rt_mode(void)
{
   tu_foveate_rt_init();
   return rt_cfg.mode;
}

void
tu_foveate_rt_fill(struct tu_device *dev, struct tu_rt_fov_block *blk)
{
   tu_foveate_rt_init();
   memset(blk, 0, sizeof(*blk));
   blk->enable = (rt_cfg.mode & TU_RT_FOV_SHARE) ? 1 : 0;
   blk->lvl_1d = rt_cfg.lvl_1d;
   blk->r0 = rt_cfg.r0;
   blk->r1 = rt_cfg.r1;
   blk->sbs = rt_cfg.sbs;
   blk->max_lvl = rt_cfg.max_lvl;
   blk->min_w = rt_cfg.min_w;
   blk->min_h = rt_cfg.min_h;
   blk->sbs_max = cfg.sbs_max;
   blk->eye_min = cfg.eye_min;
   blk->eye_max = cfg.eye_max;
   for (unsigned e = 0; e < 2; e++)
      tu_foveate_gaze_dev(dev, e, &blk->gaze[e][0], &blk->gaze[e][1]);
}

uint32_t
tu_foveate_max_bin(void)
{
   tu_foveate_init();
   return cfg.max_bin;
}

static bool
tu_foveate_dims_ok(uint32_t w, uint32_t h)
{
   if (cfg.dims_count) {
      for (uint32_t i = 0; i < cfg.dims_count; i++)
         if (cfg.dims[i][0] == w && cfg.dims[i][1] == h)
            return true;
      return false;
   }
   if (w < cfg.min_w || h < cfg.min_h)
      return false;
   float aspect = (float) w / (float) h;
   return (aspect >= cfg.sbs_min && aspect <= cfg.sbs_max) ||
          (aspect >= cfg.eye_min && aspect <= cfg.eye_max);
}

bool
tu_foveate_dynamic_wanted(const VkRenderingInfo *info)
{
   if (!tu_foveate_enabled())
      return true;
   bool color = false;
   for (uint32_t i = 0; i < info->colorAttachmentCount; i++)
      color |= info->pColorAttachments[i].imageView != VK_NULL_HANDLE;
   if (!color)
      return false;
   if (!info->viewMask && info->layerCount > 1)
      return false;
   return tu_foveate_dims_ok(info->renderArea.offset.x + info->renderArea.extent.width,
                             info->renderArea.offset.y + info->renderArea.extent.height);
}

bool
tu_foveate_pass_eligible(struct tu_cmd_buffer *cmd)
{
   return tu_foveate_fb_eligible(cmd->state.pass, cmd->state.framebuffer);
}

bool
tu_foveate_fb_eligible(const struct tu_render_pass *pass,
                       const struct tu_framebuffer *fb)
{
   if (!tu_foveate_enabled())
      return false;
   if (!pass || !fb || !pass->has_fdm ||
       pass->fragment_density_map.attachment != VK_ATTACHMENT_UNUSED)
      return false;
   if (!tu_foveate_dims_ok(fb->width, fb->height))
      return false;
   for (uint32_t i = 0; i < pass->subpass_count; i++)
      if (pass->subpasses[i].color_count)
         return true;
   return false;
}

bool
tu_foveate_wants_gmem(struct tu_cmd_buffer *cmd)
{
   if (!tu_foveate_pass_eligible(cmd))
      return false;
   if (cfg.veto && cmd->state.rp.foveate_veto) {
      p_atomic_inc(&stat_vetoed);
      return false;
   }
   return cmd->state.rp.drawcall_count >= cfg.min_draws;
}

void
tu_foveate_frag_area(struct tu_cmd_buffer *cmd,
                     uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                     unsigned view, struct tu_frag_area *area)
{
   area->width = area->height = 1.0f;
   if (!tu_foveate_pass_eligible(cmd))
      return;
   if (cfg.veto && cmd->state.rp.foveate_veto)
      return;

   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   uint32_t w = fb->width, h = fb->height;
   x2 = MIN2(x2, w);
   y2 = MIN2(y2, h);
   if (x1 >= x2 || y1 >= y2)
      return;

   bool sbs = (float) w >= cfg.sbs_min * (float) h;
   float ew = sbs ? w * 0.5f : (float) w;
   float eh = (float) h;
   unsigned eye;
   float ex0;
   if (sbs) {
      eye = (x1 + x2) / 2 >= w / 2;
      ex0 = eye ? ew : 0.f;
   } else {
      eye = view & 1;
      ex0 = 0.f;
   }

   float gx, gy;
   tu_foveate_gaze(cmd, eye, &gx, &gy);
   float px = ex0 + gx * ew, py = gy * eh;
   float nx = CLAMP(px, (float) x1, (float) x2), ny = CLAMP(py, (float) y1, (float) y2);
   float d = sqrtf((nx - px) * (nx - px) + (ny - py) * (ny - py)) / eh;

   float a = d < cfg.r0 ? 1.f : d < cfg.r1 ? 2.f : 4.f;
   a = MIN2(a, (float) cfg.max_area);
   area->width = area->height = a;
   if (cfg.log_every)
      p_atomic_inc(&stat_tiles[a >= 4.f ? 2 : a >= 2.f ? 1 : 0]);
}

/*
 * TU_VRS: driver-injected attachment VRS.
 *
 * Per latch epoch (the same per-presented-frame key the FDM/RT/gaze-block
 * consumers use) the gaze, the /dev/shm/tu-vrs-ctl control word and the
 * validity state machine are evaluated once and frozen into a
 * tu_vrs_frame snapshot. Command buffers turn that snapshot into an 8x8
 * texel R8 rate map (Vulkan encoding: 0 = 1x1, 5 = 2x2, 10 = 4x4) in their
 * own sub-stream memory and point GRAS_QUALITY_BUFFER at it for eligible
 * passes. Other passes get a zero (all 1x1) map so any draw state that
 * enabled attachment FSR (secondaries cannot see pass dimensions) reads a
 * valid buffer.
 */

#define TU_VRS_MAX_DIMS 8
#define TU_VRS_CTL_MAX 256

enum tu_vrs_reason {
   TU_VRS_R_NONE,
   TU_VRS_R_CTL_OFF,
   TU_VRS_R_SHM_MISSING,
   TU_VRS_R_SHM_STALE,
   TU_VRS_R_SHM_TORN,
   TU_VRS_R_SHM_VERSION,
   TU_VRS_R_MODE_UNKNOWN,
   TU_VRS_R_INVALID,
   TU_VRS_R_HOLD_EXPIRED,
   TU_VRS_R_HYSTERESIS,
   TU_VRS_R_NO_EYE,
   TU_VRS_R_CTL_PARSE,
   TU_VRS_R_COUNT,
};

static const char *const tu_vrs_reason_names[TU_VRS_R_COUNT] = {
   "none", "ctl-off", "shm-missing", "shm-stale", "shm-torn", "shm-version",
   "mode-unknown", "invalid", "hold-expired", "hysteresis", "no-eye",
   "ctl-parse",
};

static const char *const tu_vrs_state_names[3] = { "off", "active", "held" };

struct tu_vrs_ctl {
   bool on;
   char layout;
   float r0_deg, r1_deg;  /* <0: layout default */
   float r0_eh, r1_eh;    /* <0: layout default */
   uint32_t max;          /* 1, 2, 4 */
   bool harsh;
   bool parse_error;
   /* TU_VRS_TRANSL scope (only parsed when TU_VRS_TRANSL is set) */
   bool t_on;
   float t_r0_deg, t_r1_deg;  /* <0: base radii */
   float t_r0_eh, t_r1_eh;    /* <0: base radii */
   uint32_t t_max;            /* 1, 2, 4 */
};

/* One TU_VRS_PS_ALLOW/DENY entry: the top <bits> of the 64-bit DXIL hash
 * (8..16 hex digits given) and the coarsest level allowed for it
 * (0 = 1x1, 1 = 2x2, 2 = 4x4; denylist entries ignore it).
 */
struct tu_vrs_ps_entry {
   uint64_t val;
   uint8_t bits;
   uint8_t cap;
};

struct tu_vrs_ps_list {
   bool set;          /* the variable was given (allow: enables allow mode) */
   struct tu_vrs_ps_entry *e;
   uint32_t n, alloc;
};

struct tu_vrs_cfg {
   bool enabled;
   uint32_t dims[TU_VRS_MAX_DIMS][2];
   uint32_t dims_count;
   struct tu_vrs_ctl env;  /* defaults from the environment */
   uint32_t dither;
   uint64_t hold_ns, hyst_ns;
   uint32_t ramp;
   float saccade_scale;
   float sbs_min;
   bool linear;
   uint32_t log_every;
   const char *ctl_path;
   const char *dump;
   uint32_t dump_max;
   bool require_v2;
   /* TU_VRS_PS_ALLOW / TU_VRS_PS_DENY / TU_VRS_PS_MODE / TU_VRS_PS_DIMS */
   struct tu_vrs_ps_list ps_allow, ps_deny;
   bool ps_deny_builtin;
   bool ps_strict;      /* allowlist decides in every pass, not only depth-less */
   bool ps_error;       /* a list failed to parse: no draw shades coarse */
   uint32_t ps_log;
   uint32_t ps_dims[TU_VRS_MAX_DIMS][2];
   uint32_t ps_dims_count;
   /* TU_VRS_APP_COMB=max: an app attachment combiner of MAX or MUL is
    * treated like KEEP. The pass has no app rate attachment, so the app's
    * attachment rate is 1x1 and both ops are the identity there.
    */
   bool app_comb_max;
   /* TU_VRS_TRANSL: render-area dims of the separate-translucency class.
    * Depth passes on this list (and not on TU_VRS_DIMS) get the gaze map
    * with the transl radii; only draws that blend into a written colour
    * attachment shade coarse there. Unset: no pass is in this class.
    */
   uint32_t transl_dims[TU_VRS_MAX_DIMS][2];
   uint32_t transl_dims_count;
};

static struct tu_vrs_cfg vrs_cfg;

struct tu_vrs_latch {
   uint64_t key;
   struct tu_vrs_frame frame;
   /* validity state machine (shm source only) */
   bool active;
   int64_t valid_since_ns;
   int64_t last_valid_ns;
   uint64_t ramp_start;
   float held_uv[2][2];
   float held_scale[2][2];
   bool held_eye[2];
   uint32_t held_flags;
   /* control word */
   char ctl_raw[TU_VRS_CTL_MAX];
   bool ctl_seen;
   bool ctl_present;
   struct tu_vrs_ctl ctl;
   uint32_t ctl_changes;
   uint32_t last_state, last_reason;
   bool logged_once[TU_VRS_R_COUNT];
   /* per-log-interval latch age stats */
   int64_t age_sum_ns, age_max_ns;
   uint32_t age_n;
   uint32_t epochs, epochs_active;
};

static struct tu_vrs_latch vrs_latch = { UINT64_MAX };

static uint32_t vrs_stat_pass[TU_VRS_PASS_KINDS];
static uint32_t vrs_stat_texels[3];
static uint32_t vrs_stat_maps;
static uint32_t vrs_stat_state[4]; /* injected, capped(discard), vetoed, app-nontrivial */
/* App attachment combiner (op1) seen at the app-combiner check, by
 * VkFragmentShadingRateCombinerOpKHR value 0..4, and how many MAX/MUL draws
 * were accepted under TU_VRS_APP_COMB=max.
 */
static uint32_t vrs_stat_appop[5];
static uint32_t vrs_stat_appop_accepted;
/* PS-list decisions: allowlisted, denied, not listed (skipped), no DXIL hash
 * (skipped in allow mode), capped by an allowlist entry; depth-less passes
 * that received the gaze map.
 */
enum { VRS_PS_ALLOW, VRS_PS_DENY, VRS_PS_UNLISTED, VRS_PS_UNHASHED,
       VRS_PS_CAPPED, VRS_PS_DEPTHLESS_PASS, VRS_PS_STATS };
static uint32_t vrs_stat_ps[VRS_PS_STATS];
static uint32_t vrs_dumped;
/* TU_VRS_TRANSL: transl-class passes with the gaze map / flat (off), and
 * draws kept full rate because nothing blends.
 */
enum { VRS_TR_GAZE, VRS_TR_OFF, VRS_TR_UNBLENDED, VRS_TR_STATS };
static uint32_t vrs_stat_tr[VRS_TR_STATS];

static const float tu_vrs_layout_eh[3][2] = {
   { 0.20f, INFINITY }, /* A */
   { 0.15f, 0.30f },    /* B */
   { 0.10f, 0.22f },    /* C */
};
static const float tu_vrs_layout_deg[3][2] = {
   { 15.0f, INFINITY }, /* A */
   { 13.0f, 24.0f },    /* B */
   { 10.0f, 18.0f },    /* C */
};

static unsigned
tu_vrs_layout_index(char l)
{
   return l == 'A' ? 0 : l == 'C' ? 2 : 1;
}

static void
tu_vrs_parse_dims_into(const char *s, uint32_t (*dims)[2], uint32_t *count)
{
   while (s && *s && *count < TU_VRS_MAX_DIMS) {
      uint32_t w, h;
      if (sscanf(s, "%ux%u", &w, &h) != 2)
         break;
      dims[*count][0] = w;
      dims[*count][1] = h;
      (*count)++;
      s = strchr(s, ',');
      if (s)
         s++;
   }
}

static void
tu_vrs_parse_dims(const char *s)
{
   tu_vrs_parse_dims_into(s, vrs_cfg.dims, &vrs_cfg.dims_count);
}

/* ---- TU_VRS_PS_ALLOW / TU_VRS_PS_DENY ----
 * Value: an inline list, or "@<file>" / an absolute path to a file holding
 * one. Tokens are separated by whitespace, ',' or ';'; '#' starts a comment
 * that runs to the end of the line. A token is
 *   <8..16 hex digits>[:1|2|4]   DXIL hash prefix, optional coarsest rate
 *   default                      (deny only) the built-in list (velocity,
 *                                FSR reactive/composition inputs, HUD/UI)
 *   none                         (deny only) no built-in list
 * TU_VRS_PS_DENY unset means "default". Any malformed token is a parse
 * error: logged, and then no draw shades coarse at all (fail closed).
 */
#define TU_VRS_PS_FILE_MAX (1u << 20)

static bool
tu_vrs_ps_add(struct tu_vrs_ps_list *l, uint64_t val, uint8_t bits, uint8_t cap)
{
   if (l->n == l->alloc) {
      uint32_t na = l->alloc ? l->alloc * 2 : 16;
      void *ne = realloc(l->e, na * sizeof(*l->e));
      if (!ne)
         return false;
      l->e = (struct tu_vrs_ps_entry *) ne;
      l->alloc = na;
   }
   l->e[l->n++] = (struct tu_vrs_ps_entry) { val, bits, cap };
   return true;
}

static bool
tu_vrs_ps_parse_token(const char *t, struct tu_vrs_ps_list *l, bool deny)
{
   if (deny && !strcmp(t, "none")) {
      vrs_cfg.ps_deny_builtin = false;
      return true;
   }
   if (deny && !strcmp(t, "default")) {
      vrs_cfg.ps_deny_builtin = true;
      return true;
   }
   uint64_t v = 0;
   unsigned n = 0;
   for (; t[n] && t[n] != ':'; n++) {
      char c = t[n];
      unsigned d;
      if (c >= '0' && c <= '9')
         d = c - '0';
      else if (c >= 'a' && c <= 'f')
         d = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
         d = c - 'A' + 10;
      else
         return false;
      if (n >= 16)
         return false;
      v = (v << 4) | d;
   }
   if (n < 8)
      return false;
   uint8_t cap = 2;
   if (t[n] == ':') {
      if (deny || !t[n + 1] || t[n + 2])
         return false;
      switch (t[n + 1]) {
      case '1': cap = 0; break;
      case '2': cap = 1; break;
      case '4': cap = 2; break;
      default: return false;
      }
   }
   return tu_vrs_ps_add(l, v, (uint8_t) (n * 4), cap);
}

static bool
tu_vrs_ps_parse(const char *name, const char *spec, struct tu_vrs_ps_list *l,
                bool deny)
{
   char *text = NULL;
   if (spec[0] == '@' || spec[0] == '/') {
      const char *path = spec[0] == '@' ? spec + 1 : spec;
      FILE *fp = fopen(path, "rb");
      if (!fp) {
         mesa_loge("tu-vrs-ps: %s: cannot open %s", name, path);
         return false;
      }
      text = (char *) calloc(1, TU_VRS_PS_FILE_MAX + 1);
      size_t got = text ? fread(text, 1, TU_VRS_PS_FILE_MAX + 1, fp) : 0;
      fclose(fp);
      if (!text || got > TU_VRS_PS_FILE_MAX) {
         mesa_loge("tu-vrs-ps: %s: %s unreadable or over %u bytes", name, path,
                   TU_VRS_PS_FILE_MAX);
         free(text);
         return false;
      }
      text[got] = 0;
   } else {
      text = strdup(spec);
      if (!text)
         return false;
   }
   bool ok = true;
   char *p = text;
   while (*p) {
      if (*p == '#') {
         while (*p && *p != '\n')
            p++;
         continue;
      }
      if (strchr(" \t\r\n,;", *p)) {
         p++;
         continue;
      }
      char *t = p;
      while (*p && !strchr(" \t\r\n,;#", *p))
         p++;
      char save = *p;
      *p = 0;
      if (!tu_vrs_ps_parse_token(t, l, deny)) {
         mesa_loge("tu-vrs-ps: %s: bad token '%s'", name, t);
         ok = false;
      }
      *p = save;
   }
   free(text);
   return ok;
}

static bool
tu_vrs_ps_match(const struct tu_vrs_ps_list *l, uint64_t h, uint8_t *cap)
{
   for (uint32_t i = 0; i < l->n; i++) {
      const struct tu_vrs_ps_entry *e = &l->e[i];
      if ((h >> (64 - e->bits)) == e->val) {
         if (cap)
            *cap = e->cap;
         return true;
      }
   }
   return false;
}

static bool
tu_vrs_ps_builtin_denied(uint64_t h)
{
   uint32_t lo = 0, hi = ARRAY_SIZE(tu_vrs_ps_deny_builtin);
   while (lo < hi) {
      uint32_t mid = (lo + hi) / 2;
      if (tu_vrs_ps_deny_builtin[mid] == h)
         return true;
      if (tu_vrs_ps_deny_builtin[mid] < h)
         lo = mid + 1;
      else
         hi = mid;
   }
   return false;
}

static bool
tu_vrs_ps_denied(uint64_t h)
{
   if (!h)
      return false;
   return (vrs_cfg.ps_deny_builtin && tu_vrs_ps_builtin_denied(h)) ||
          tu_vrs_ps_match(&vrs_cfg.ps_deny, h, NULL);
}

static void
tu_vrs_ps_init(void)
{
   vrs_cfg.ps_deny_builtin = true;
   const char *allow = os_get_option("TU_VRS_PS_ALLOW");
   const char *deny = os_get_option("TU_VRS_PS_DENY");
   if (allow && *allow) {
      vrs_cfg.ps_allow.set = true;
      if (!tu_vrs_ps_parse("TU_VRS_PS_ALLOW", allow, &vrs_cfg.ps_allow, false))
         vrs_cfg.ps_error = true;
   }
   if (deny && *deny) {
      vrs_cfg.ps_deny.set = true;
      if (!tu_vrs_ps_parse("TU_VRS_PS_DENY", deny, &vrs_cfg.ps_deny, true))
         vrs_cfg.ps_error = true;
   }
   const char *mode = os_get_option("TU_VRS_PS_MODE");
   vrs_cfg.ps_strict = true;
   if (mode && !strcmp(mode, "extend"))
      vrs_cfg.ps_strict = false;
   else if (mode && strcmp(mode, "strict")) {
      mesa_loge("tu-vrs-ps: TU_VRS_PS_MODE=%s unknown (strict|extend)", mode);
      vrs_cfg.ps_error = true;
   }
   tu_vrs_parse_dims_into(os_get_option("TU_VRS_PS_DIMS"), vrs_cfg.ps_dims,
                          &vrs_cfg.ps_dims_count);
   const char *comb = os_get_option("TU_VRS_APP_COMB");
   if (comb && !strcmp(comb, "max"))
      vrs_cfg.app_comb_max = true;
   else if (comb && *comb && strcmp(comb, "keep"))
      mesa_loge("tu-vrs: TU_VRS_APP_COMB=%s unknown (keep|max), using keep", comb);
   if (comb && *comb)
      mesa_logi("tu-vrs: app_comb=%s", vrs_cfg.app_comb_max ? "max" : "keep");
   vrs_cfg.ps_log = debug_get_num_option("TU_VRS_PS_LOG", 0);
   if (vrs_cfg.ps_error)
      mesa_loge("tu-vrs-ps: list/mode parse error: no draw will shade coarse");

   mesa_logi("tu-vrs-ps: allow=%s entries=%u mode=%s deny builtin=%d(%u) extra=%u ps_dims=%u error=%d log=%u",
             vrs_cfg.ps_allow.set ? "on" : "off", vrs_cfg.ps_allow.n,
             vrs_cfg.ps_strict ? "strict" : "extend", vrs_cfg.ps_deny_builtin,
             (unsigned) ARRAY_SIZE(tu_vrs_ps_deny_builtin), vrs_cfg.ps_deny.n,
             vrs_cfg.ps_dims_count, vrs_cfg.ps_error, vrs_cfg.ps_log);
   for (uint32_t i = 0; i < vrs_cfg.ps_allow.n; i++) {
      const struct tu_vrs_ps_entry *e = &vrs_cfg.ps_allow.e[i];
      /* Report allow entries a deny entry shadows (deny wins). */
      bool shadowed = false;
      for (uint32_t j = 0; j < ARRAY_SIZE(tu_vrs_ps_deny_builtin) &&
                           vrs_cfg.ps_deny_builtin && !shadowed; j++)
         shadowed = (tu_vrs_ps_deny_builtin[j] >> (64 - e->bits)) == e->val;
      for (uint32_t j = 0; j < vrs_cfg.ps_deny.n && !shadowed; j++) {
         const struct tu_vrs_ps_entry *d = &vrs_cfg.ps_deny.e[j];
         uint8_t b = MIN2(d->bits, e->bits);
         shadowed = (d->val >> (d->bits - b)) == (e->val >> (e->bits - b));
      }
      mesa_logi("tu-vrs-ps: allow[%u]=%0*" PRIx64 " cap=%ux%u%s", i,
                (int) (e->bits / 4), e->val, 1u << e->cap, 1u << e->cap,
                shadowed ? " SHADOWED-BY-DENY" : "");
   }
   for (uint32_t i = 0; i < vrs_cfg.ps_dims_count; i++)
      mesa_logi("tu-vrs-ps: ps_dims[%u]=%ux%u", i, vrs_cfg.ps_dims[i][0],
                vrs_cfg.ps_dims[i][1]);
}

/* Control word / ctl file grammar (whitespace or comma separated tokens):
 *   on | off | 1 | 0          enable / disable the gaze map (flat when off)
 *   layout=A|B|C              ring layout
 *   r0=<deg> r1=<deg>         radii in degrees (used when gaze scale published)
 *   r0eh=<f> r1eh=<f>         radii in eye-heights (fallback)
 *   max=1|2|4                 coarsest rate allowed
 *   harsh | normal            debug: 4x4 straight outside a small fovea
 * With TU_VRS_TRANSL set (otherwise these are parse errors, as before):
 *   transl=on|off             gaze map on transl-class passes (independent
 *                             of on/off, which governs TU_VRS_DIMS passes)
 *   tr0=<deg> tr1=<deg>       transl radii in degrees
 *   tr0eh=<f> tr1eh=<f>       transl radii in eye-heights
 *   tmax=1|2|4                coarsest rate on transl-class passes
 * An unknown token is a parse error and forces the gaze map off.
 */
static bool
tu_vrs_parse_transl_token(const char *t, struct tu_vrs_ctl *c)
{
   float f;
   unsigned u;
   if (!strcmp(t, "transl=on") || !strcmp(t, "transl=1"))
      c->t_on = true;
   else if (!strcmp(t, "transl=off") || !strcmp(t, "transl=0"))
      c->t_on = false;
   else if (sscanf(t, "tr0=%f", &f) == 1 && f > 0.f)
      c->t_r0_deg = f;
   else if (sscanf(t, "tr1=%f", &f) == 1 && f > 0.f)
      c->t_r1_deg = f;
   else if (sscanf(t, "tr0eh=%f", &f) == 1 && f > 0.f)
      c->t_r0_eh = f;
   else if (sscanf(t, "tr1eh=%f", &f) == 1 && f > 0.f)
      c->t_r1_eh = f;
   else if (sscanf(t, "tmax=%u", &u) == 1 && (u == 1 || u == 2 || u == 4))
      c->t_max = u;
   else
      return false;
   return true;
}

static void
tu_vrs_parse_ctl(const char *s, struct tu_vrs_ctl *c)
{
   char buf[TU_VRS_CTL_MAX];
   snprintf(buf, sizeof(buf), "%s", s);
   char *save = NULL;
   for (char *t = strtok_r(buf, " \t\r\n,", &save); t;
        t = strtok_r(NULL, " \t\r\n,", &save)) {
      float f;
      unsigned u;
      if (!strcmp(t, "on") || !strcmp(t, "1"))
         c->on = true;
      else if (!strcmp(t, "off") || !strcmp(t, "0"))
         c->on = false;
      else if (!strcmp(t, "harsh"))
         c->harsh = true;
      else if (!strcmp(t, "normal"))
         c->harsh = false;
      else if (!strncmp(t, "layout=", 7) &&
               (t[7] == 'A' || t[7] == 'B' || t[7] == 'C') && !t[8])
         c->layout = t[7];
      else if (sscanf(t, "r0=%f", &f) == 1 && f > 0.f)
         c->r0_deg = f;
      else if (sscanf(t, "r1=%f", &f) == 1 && f > 0.f)
         c->r1_deg = f;
      else if (sscanf(t, "r0eh=%f", &f) == 1 && f > 0.f)
         c->r0_eh = f;
      else if (sscanf(t, "r1eh=%f", &f) == 1 && f > 0.f)
         c->r1_eh = f;
      else if (sscanf(t, "max=%u", &u) == 1 && (u == 1 || u == 2 || u == 4))
         c->max = u;
      else if (vrs_cfg.transl_dims_count && tu_vrs_parse_transl_token(t, c))
         ;
      else
         c->parse_error = true;
   }
}

static void
tu_vrs_init_once(void)
{
   tu_foveate_init();
   bool want = debug_get_bool_option("TU_VRS", false);
   if (want && cfg.enabled) {
      mesa_loge("tu-vrs: TU_VRS and TU_FOVEATE are mutually exclusive; TU_VRS disabled");
      want = false;
   }
   vrs_cfg.enabled = want;
   if (!want)
      return;

   const char *dims = os_get_option("TU_VRS_DIMS");
   /* Scene render sizes: today's 1712x960 SBS (856x960/eye) and FSR
    * Performance 1280x720 SBS (640x720/eye).
    */
   tu_vrs_parse_dims(dims ? dims : "1712x960,856x960,1280x720,640x720");
   tu_vrs_ps_init();

   struct tu_vrs_ctl *e = &vrs_cfg.env;
   e->on = true;
   const char *layout = os_get_option("TU_VRS_LAYOUT");
   e->layout = (layout && (layout[0] == 'A' || layout[0] == 'C')) ? layout[0] : 'B';
   e->r0_deg = env_f("TU_VRS_R0_DEG", -1.f);
   e->r1_deg = env_f("TU_VRS_R1_DEG", -1.f);
   e->r0_eh = env_f("TU_VRS_R0", -1.f);
   e->r1_eh = env_f("TU_VRS_R1", -1.f);
   uint32_t max = debug_get_num_option("TU_VRS_MAX", 4);
   e->max = max >= 4 ? 4 : max >= 2 ? 2 : 1;
   const char *dbg = os_get_option("TU_VRS_DEBUG");
   e->harsh = dbg && !strcmp(dbg, "harsh");

   const char *transl = os_get_option("TU_VRS_TRANSL");
   if (transl && *transl) {
      tu_vrs_parse_dims_into(transl, vrs_cfg.transl_dims,
                             &vrs_cfg.transl_dims_count);
      e->t_on = debug_get_bool_option("TU_VRS_TRANSL_ON", true);
      e->t_r0_deg = env_f("TU_VRS_TRANSL_R0_DEG", -1.f);
      e->t_r1_deg = env_f("TU_VRS_TRANSL_R1_DEG", -1.f);
      e->t_r0_eh = env_f("TU_VRS_TRANSL_R0", -1.f);
      e->t_r1_eh = env_f("TU_VRS_TRANSL_R1", -1.f);
      uint32_t tmax = debug_get_num_option("TU_VRS_TRANSL_MAX", 2);
      e->t_max = tmax >= 4 ? 4 : tmax >= 2 ? 2 : 1;
      if (!vrs_cfg.transl_dims_count)
         mesa_loge("tu-vrs: TU_VRS_TRANSL='%s' has no WxH entry; transl scope off", transl);
   }

   vrs_cfg.dither = MIN2(debug_get_num_option("TU_VRS_DITHER", 2), 8);
   vrs_cfg.hold_ns = (uint64_t) debug_get_num_option("TU_VRS_HOLD_MS", 300) * 1000000ull;
   vrs_cfg.hyst_ns = (uint64_t) debug_get_num_option("TU_VRS_HYST_MS", 100) * 1000000ull;
   vrs_cfg.ramp = debug_get_num_option("TU_VRS_RAMP", 4);
   vrs_cfg.saccade_scale = env_f("TU_VRS_SACCADE_SCALE", 1.5f);
   vrs_cfg.sbs_min = cfg.sbs_min;
   vrs_cfg.linear = debug_get_bool_option("TU_VRS_LINEAR", false);
   vrs_cfg.log_every = debug_get_num_option("TU_VRS_LOG", 0);
   vrs_cfg.ctl_path = os_get_option("TU_VRS_CTL");
   if (!vrs_cfg.ctl_path)
      vrs_cfg.ctl_path = "/dev/shm/tu-vrs-ctl";
   if (!strcmp(vrs_cfg.ctl_path, "none"))
      vrs_cfg.ctl_path = NULL;
   vrs_cfg.dump = os_get_option("TU_VRS_DUMP");
   vrs_cfg.dump_max = debug_get_num_option("TU_VRS_DUMP_MAX", 16);
   vrs_cfg.require_v2 = debug_get_bool_option("TU_VRS_REQUIRE_V2", true);

   mesa_logi("tu-vrs: enabled dims=%u layout=%c r0_deg=%.2f r1_deg=%.2f r0_eh=%.3f r1_eh=%.3f max=%u harsh=%d dither=%u hold=%ums hyst=%ums ramp=%u saccade_scale=%.2f linear=%d log=%u ctl=%s gaze=%d fixed=%.3f,%.3f stale=%ums require_v2=%d",
             vrs_cfg.dims_count, e->layout, e->r0_deg, e->r1_deg, e->r0_eh, e->r1_eh,
             e->max, e->harsh, vrs_cfg.dither, (unsigned) (vrs_cfg.hold_ns / 1000000ull),
             (unsigned) (vrs_cfg.hyst_ns / 1000000ull), vrs_cfg.ramp, vrs_cfg.saccade_scale,
             vrs_cfg.linear, vrs_cfg.log_every,
             vrs_cfg.ctl_path ? vrs_cfg.ctl_path : "(none)", cfg.src, cfg.fixed_x,
             cfg.fixed_y, (unsigned) (cfg.stale_ns / 1000000ull), vrs_cfg.require_v2);
   for (uint32_t i = 0; i < vrs_cfg.dims_count; i++)
      mesa_logi("tu-vrs: dims[%u]=%ux%u", i, vrs_cfg.dims[i][0], vrs_cfg.dims[i][1]);
   if (vrs_cfg.transl_dims_count) {
      mesa_logi("tu-vrs-transl: on=%d r0_deg=%.2f r1_deg=%.2f r0_eh=%.3f r1_eh=%.3f max=%u (<0: base radii)",
                e->t_on, e->t_r0_deg, e->t_r1_deg, e->t_r0_eh, e->t_r1_eh, e->t_max);
      for (uint32_t i = 0; i < vrs_cfg.transl_dims_count; i++)
         mesa_logi("tu-vrs-transl: dims[%u]=%ux%u%s", i, vrs_cfg.transl_dims[i][0],
                   vrs_cfg.transl_dims[i][1],
                   tu_vrs_dims_ok(vrs_cfg.transl_dims[i][0], vrs_cfg.transl_dims[i][1]) ?
                   " (also on TU_VRS_DIMS: base wins)" : "");
   }
}

static void
tu_vrs_init(void)
{
   static once_flag once = ONCE_FLAG_INIT;
   call_once(&once, tu_vrs_init_once);
}

bool
tu_vrs_enabled(void)
{
   tu_vrs_init();
   return vrs_cfg.enabled;
}

bool
tu_vrs_linear(void)
{
   return vrs_cfg.linear;
}

bool
tu_vrs_ps_allow_mode(void)
{
   return vrs_cfg.ps_allow.set && !vrs_cfg.ps_error;
}

bool
tu_vrs_ps_dims_ok(uint32_t w, uint32_t h)
{
   for (uint32_t i = 0; i < vrs_cfg.ps_dims_count; i++)
      if (vrs_cfg.ps_dims[i][0] == w && vrs_cfg.ps_dims[i][1] == h)
         return true;
   return tu_vrs_dims_ok(w, h);
}

bool
tu_vrs_dims_ok(uint32_t w, uint32_t h)
{
   for (uint32_t i = 0; i < vrs_cfg.dims_count; i++)
      if (vrs_cfg.dims[i][0] == w && vrs_cfg.dims[i][1] == h)
         return true;
   return false;
}

bool
tu_vrs_transl_dims_ok(uint32_t w, uint32_t h)
{
   for (uint32_t i = 0; i < vrs_cfg.transl_dims_count; i++)
      if (vrs_cfg.transl_dims[i][0] == w && vrs_cfg.transl_dims[i][1] == h)
         return true;
   return false;
}

/* A transl-class draw shades coarse only if it blends into at least one
 * written colour attachment (translucent materials); opaque writes in the
 * same pass class (scene-colour copies, clears) stay full rate.
 */
bool
tu_vrs_transl_draw_ok(const struct vk_color_blend_state *cb)
{
   for (uint32_t i = 0; i < cb->attachment_count && i < MESA_VK_MAX_COLOR_ATTACHMENTS; i++) {
      if ((cb->color_write_enables & (1u << i)) &&
          cb->attachments[i].blend_enable && cb->attachments[i].write_mask)
         return true;
   }
   p_atomic_inc(&vrs_stat_tr[VRS_TR_UNBLENDED]);
   return false;
}

bool
tu_vrs_frame_select(struct tu_vrs_frame *f, bool transl)
{
   if (!transl)
      return f->base_on;
   if (!f->t_on) {
      p_atomic_inc(&vrs_stat_tr[VRS_TR_OFF]);
      return false;
   }
   f->r0 = f->t_r0;
   f->r1 = f->t_r1;
   f->cap = f->t_cap;
   p_atomic_inc(&vrs_stat_tr[VRS_TR_GAZE]);
   return true;
}

/* Read the ctl file once per epoch. Missing file: environment defaults. */
static void
tu_vrs_read_ctl(struct tu_vrs_latch *l)
{
   char raw[TU_VRS_CTL_MAX] = { 0 };
   bool present = false;
   if (vrs_cfg.ctl_path) {
      int fd = open(vrs_cfg.ctl_path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
      if (fd >= 0) {
         ssize_t n = read(fd, raw, sizeof(raw) - 1);
         close(fd);
         present = true;
         raw[n > 0 ? n : 0] = 0;
      }
   }
   if (l->ctl_seen && present == l->ctl_present && !strcmp(raw, l->ctl_raw))
      return;

   struct tu_vrs_ctl c = vrs_cfg.env;
   c.parse_error = false;
   if (present)
      tu_vrs_parse_ctl(raw, &c);
   l->ctl_seen = true;
   l->ctl_present = present;
   memcpy(l->ctl_raw, raw, sizeof(raw));
   l->ctl = c;
   l->ctl_changes++;
   for (char *p = raw; *p; p++)
      if (*p == '\n' || *p == '\r')
         *p = ' ';
   mesa_logi("tu-vrs: ctl change=%u epoch=%" PRIu64 " present=%d raw='%s' -> on=%d layout=%c r0_deg=%.2f r1_deg=%.2f r0_eh=%.3f r1_eh=%.3f max=%u harsh=%d parse_error=%d",
             l->ctl_changes, l->key, present, raw, c.on, c.layout, c.r0_deg,
             c.r1_deg, c.r0_eh, c.r1_eh, c.max, c.harsh, c.parse_error);
   if (vrs_cfg.transl_dims_count)
      mesa_logi("tu-vrs-transl: ctl change=%u -> transl=%d tr0_deg=%.2f tr1_deg=%.2f tr0_eh=%.3f tr1_eh=%.3f tmax=%u",
                l->ctl_changes, c.t_on, c.t_r0_deg, c.t_r1_deg, c.t_r0_eh,
                c.t_r1_eh, c.t_max);
}

struct tu_vrs_sample {
   uint32_t reason;     /* TU_VRS_R_NONE when ok */
   bool transport_ok;   /* fresh, consistent, understood shm */
   bool eye[2];
   float uv[2][2];
   float scale[2][2];
   uint32_t flags;
   int64_t age_ns;
};

/* Gaze shm v2: layout after the v1 prefix (see tu_gaze_shm_v2 in
 * tu_foveate.h, copied from gazed-v2/tu_gaze_shm.h).
 */
static const struct tu_gaze_shm_v2 *volatile vrs_shm;
static int64_t vrs_shm_retry_ns;

/* Separate read-only mapping that checks the file covers the v2 struct, so a
 * short file can never SIGBUS. Retried at most once per second.
 */
static const struct tu_gaze_shm_v2 *
tu_vrs_map_shm(int64_t now)
{
   const struct tu_gaze_shm_v2 *p = vrs_shm;
   if (p)
      return p;
   if (now < vrs_shm_retry_ns)
      return NULL;
   vrs_shm_retry_ns = now + 1000000000ll;
   int fd = open(cfg.path, O_RDONLY | O_CLOEXEC);
   if (fd < 0)
      return NULL;
   struct stat st;
   if (fstat(fd, &st) || st.st_size < (off_t) sizeof(struct tu_gaze_shm_v2)) {
      close(fd);
      return NULL;
   }
   void *m = mmap(NULL, sizeof(struct tu_gaze_shm_v2), PROT_READ, MAP_SHARED, fd, 0);
   close(fd);
   if (m == MAP_FAILED)
      return NULL;
   p = (const struct tu_gaze_shm_v2 *) m;
   if (p->v1.magic != TU_GAZE_MAGIC) {
      munmap(m, sizeof(struct tu_gaze_shm_v2));
      return NULL;
   }
   mesa_logi("tu-vrs: gaze shm %s mapped (version field %u)", cfg.path, p->version);
   vrs_shm = p;
   return p;
}

static void
tu_vrs_read_shm(int64_t now, struct tu_vrs_sample *s)
{
   const struct tu_gaze_shm_v2 *p2 = tu_vrs_map_shm(now);
   if (!p2) {
      s->reason = TU_VRS_R_SHM_MISSING;
      return;
   }
   struct tu_gaze_shm_v2 c;
   bool consistent = false;
   for (unsigned tries = 0; tries < 4 && !consistent; tries++) {
      uint32_t s0 = p_atomic_read(&p2->v1.seq);
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      memcpy(&c, (const void *) p2, sizeof(c));
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      consistent = !(s0 & 1) && s0 == p_atomic_read(&p2->v1.seq);
   }
   if (!consistent) {
      s->reason = TU_VRS_R_SHM_TORN;
      return;
   }
   if (c.v1.magic != TU_GAZE_MAGIC) {
      s->reason = TU_VRS_R_SHM_MISSING;
      return;
   }
   if ((uint64_t) now - c.v1.time_ns > cfg.stale_ns) {
      s->reason = TU_VRS_R_SHM_STALE;
      return;
   }
   bool v2 = c.version == TU_GAZE_SHM_VERSION &&
             (c.v1.layout & TU_GAZE_LAYOUT_VER_MASK) == TU_GAZE_LAYOUT_V2;
   if (!v2 && vrs_cfg.require_v2) {
      s->reason = TU_VRS_R_SHM_VERSION;
      return;
   }
   if (v2 && !tu_gaze_shm_mode_known(&c)) {
      s->reason = TU_VRS_R_MODE_UNKNOWN;
      return;
   }
   s->transport_ok = true;
   uint64_t st = v2 && c.sample_time_ns ? c.sample_time_ns : c.v1.time_ns;
   s->age_ns = now - (int64_t) st;
   s->flags = v2 ? c.flags : 0;
   /* A tracker sample frozen past the daemon's limit is treated like a
    * dropout (held for up to TU_VRS_HOLD_MS, then full rate).
    */
   if (!c.v1.valid || (v2 && (c.flags & TU_GAZE_FLAG_SAMPLE_FROZEN))) {
      s->reason = TU_VRS_R_INVALID;
      return;
   }
   bool any = false;
   for (unsigned e = 0; e < 2; e++) {
      float x = c.v1.uv[e][0], y = c.v1.uv[e][1];
      s->eye[e] = (!v2 || tu_gaze_shm_eye_has_fovea(&c, e)) &&
                  x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f;
      s->uv[e][0] = x;
      s->uv[e][1] = y;
      s->scale[e][0] = v2 ? c.deg_per_uv[e][0] : 0.f;
      s->scale[e][1] = v2 ? c.deg_per_uv[e][1] : 0.f;
      if (!(s->scale[e][0] > 0.f && s->scale[e][1] > 0.f &&
            s->scale[e][0] < 1000.f && s->scale[e][1] < 1000.f))
         s->scale[e][0] = s->scale[e][1] = 0.f;
      any |= s->eye[e];
   }
   s->reason = any ? TU_VRS_R_NONE : TU_VRS_R_NO_EYE;
}

static void
tu_vrs_log_transition(struct tu_vrs_latch *l)
{
   struct tu_vrs_frame *f = &l->frame;
   if (f->state == l->last_state && f->reason == l->last_reason)
      return;
   mesa_logi("tu-vrs: epoch=%" PRIu64 " state %s(%s) -> %s(%s) gaze L(%.3f,%.3f)%s R(%.3f,%.3f)%s age=%.1fms",
             l->key, tu_vrs_state_names[l->last_state],
             tu_vrs_reason_names[l->last_reason], tu_vrs_state_names[f->state],
             tu_vrs_reason_names[f->reason], f->uv[0][0], f->uv[0][1],
             f->eye[0] ? "" : "-", f->uv[1][0], f->uv[1][1],
             f->eye[1] ? "" : "-", f->sample_age_ns / 1e6);
   l->last_state = f->state;
   l->last_reason = f->reason;
}

static void
tu_vrs_log_interval(struct tu_vrs_latch *l)
{
   if (!vrs_cfg.log_every || ++l->epochs < vrs_cfg.log_every)
      return;
   const struct tu_vrs_frame *f = &l->frame;
   uint32_t pass[TU_VRS_PASS_KINDS], tex[3], st[4];
   for (unsigned i = 0; i < TU_VRS_PASS_KINDS; i++)
      pass[i] = p_atomic_xchg(&vrs_stat_pass[i], 0);
   for (unsigned i = 0; i < 3; i++)
      tex[i] = p_atomic_xchg(&vrs_stat_texels[i], 0);
   for (unsigned i = 0; i < 4; i++)
      st[i] = p_atomic_xchg(&vrs_stat_state[i], 0);
   uint32_t maps = p_atomic_xchg(&vrs_stat_maps, 0);
   mesa_logi("tu-vrs: epoch=%" PRIu64 " epochs=%u active_epochs=%u state=%s(%s) passes gaze=%u off=%u flat=%u app=%u maps=%u texels 1x1=%u 2x2=%u 4x4=%u states inj=%u cap2=%u veto=%u app_comb=%u latch_age avg=%.1fms max=%.1fms n=%u r0=%.3f r1=%.3f%s cap=%u flags=0x%x ctl_changes=%u",
             l->key, l->epochs, l->epochs_active, tu_vrs_state_names[f->state],
             tu_vrs_reason_names[f->reason], pass[TU_VRS_PASS_GAZE],
             pass[TU_VRS_PASS_OFF], pass[TU_VRS_PASS_FLAT], pass[TU_VRS_PASS_APP],
             maps, tex[0], tex[1], tex[2], st[0], st[1], st[2], st[3],
             l->age_n ? l->age_sum_ns / 1e6 / l->age_n : 0.0, l->age_max_ns / 1e6,
             l->age_n, f->r0, f->r1, f->scale[0][0] > 0.f ? "deg" : "eh", f->cap,
             f->flags, l->ctl_changes);
   if (vrs_cfg.ps_allow.set || vrs_cfg.ps_deny.set || vrs_cfg.ps_error) {
      uint32_t ps[VRS_PS_STATS];
      for (unsigned i = 0; i < VRS_PS_STATS; i++)
         ps[i] = p_atomic_xchg(&vrs_stat_ps[i], 0);
      mesa_logi("tu-vrs-ps: epoch=%" PRIu64 " decisions allow=%u deny=%u unlisted=%u unhashed=%u listcap=%u depthless_gaze_passes=%u error=%d",
                l->key, ps[VRS_PS_ALLOW], ps[VRS_PS_DENY], ps[VRS_PS_UNLISTED],
                ps[VRS_PS_UNHASHED], ps[VRS_PS_CAPPED],
                ps[VRS_PS_DEPTHLESS_PASS], vrs_cfg.ps_error);
   }
   if (vrs_cfg.transl_dims_count) {
      uint32_t tr[VRS_TR_STATS];
      for (unsigned i = 0; i < VRS_TR_STATS; i++)
         tr[i] = p_atomic_xchg(&vrs_stat_tr[i], 0);
      mesa_logi("tu-vrs-transl: epoch=%" PRIu64 " on=%d passes gaze=%u off=%u draws unblended=%u r0=%.3f r1=%.3f cap=%u",
                l->key, f->t_on, tr[VRS_TR_GAZE], tr[VRS_TR_OFF],
                tr[VRS_TR_UNBLENDED], f->t_r0, f->t_r1, f->t_cap);
   }
   {
      uint32_t op[5];
      for (unsigned i = 0; i < 5; i++)
         op[i] = p_atomic_xchg(&vrs_stat_appop[i], 0);
      uint32_t acc = p_atomic_xchg(&vrs_stat_appop_accepted, 0);
      mesa_logi("tu-vrs-app: epoch=%" PRIu64 " op1 replace=%u min=%u max=%u mul=%u accepted=%u app_comb=%s",
                l->key, op[1], op[2], op[3], op[4], acc,
                vrs_cfg.app_comb_max ? "max" : "keep");
   }
   l->epochs = l->epochs_active = 0;
   l->age_sum_ns = l->age_max_ns = 0;
   l->age_n = 0;
}

static void
tu_vrs_update(struct tu_vrs_latch *l, uint64_t frame)
{
   int64_t now = os_time_get_nano();
   tu_vrs_read_ctl(l);
   const struct tu_vrs_ctl *c = &l->ctl;

   struct tu_vrs_frame *f = &l->frame;
   memset(f, 0, sizeof(*f));
   f->epoch = frame;
   f->latch_ns = now;
   f->sample_age_ns = -1;
   f->dither = vrs_cfg.dither;

   struct tu_vrs_sample s = {};
   bool synthetic = cfg.src != TU_GAZE_SHM;
   if (synthetic) {
      float gx, gy;
      tu_foveate_gaze_synth(frame, &gx, &gy);
      s.transport_ok = true;
      for (unsigned e = 0; e < 2; e++) {
         s.eye[e] = true;
         s.uv[e][0] = gx;
         s.uv[e][1] = gy;
      }
      s.age_ns = 0;
      s.reason = TU_VRS_R_NONE;
   } else {
      tu_vrs_read_shm(now, &s);
   }

   bool use = false;
   bool t_on = vrs_cfg.transl_dims_count && c->t_on;
   if ((!c->on && !t_on) || c->parse_error) {
      f->reason = c->parse_error ? TU_VRS_R_CTL_PARSE : TU_VRS_R_CTL_OFF;
      l->active = false;
      l->valid_since_ns = 0;
   } else if (synthetic) {
      /* Synthetic gaze is always valid: no hysteresis or ramp, so a test
       * gets the full map in its first epoch.
       */
      use = true;
      if (!l->active)
         l->ramp_start = frame - vrs_cfg.ramp;
      l->active = true;
      f->state = TU_VRS_ACTIVE;
   } else if (s.reason == TU_VRS_R_NONE) {
      l->last_valid_ns = now;
      memcpy(l->held_uv, s.uv, sizeof(s.uv));
      memcpy(l->held_scale, s.scale, sizeof(s.scale));
      memcpy(l->held_eye, s.eye, sizeof(s.eye));
      l->held_flags = s.flags;
      if (!l->active) {
         if (!l->valid_since_ns)
            l->valid_since_ns = now;
         if ((uint64_t) (now - l->valid_since_ns) >= vrs_cfg.hyst_ns) {
            l->active = true;
            l->ramp_start = frame;
         }
      }
      if (l->active) {
         use = true;
         f->state = TU_VRS_ACTIVE;
      } else {
         f->reason = TU_VRS_R_HYSTERESIS;
      }
   } else if (s.transport_ok && l->active &&
              (s.reason == TU_VRS_R_INVALID || s.reason == TU_VRS_R_NO_EYE) &&
              (uint64_t) (now - l->last_valid_ns) < vrs_cfg.hold_ns) {
      /* Blink/dropout: hold the last valid gaze. */
      use = true;
      f->state = TU_VRS_HELD;
      memcpy(s.uv, l->held_uv, sizeof(s.uv));
      memcpy(s.scale, l->held_scale, sizeof(s.scale));
      memcpy(s.eye, l->held_eye, sizeof(s.eye));
      s.flags = l->held_flags;
   } else {
      f->reason = (l->active && s.transport_ok &&
                   (s.reason == TU_VRS_R_INVALID || s.reason == TU_VRS_R_NO_EYE)) ?
                  TU_VRS_R_HOLD_EXPIRED : s.reason;
      l->active = false;
      l->valid_since_ns = 0;
      if (!l->logged_once[s.reason]) {
         l->logged_once[s.reason] = true;
         mesa_logi("tu-vrs: gaze unusable (%s) at epoch %" PRIu64 "; full rate",
                   tu_vrs_reason_names[s.reason], frame);
      }
   }

   f->flags = s.flags;
   if (s.transport_ok && s.age_ns >= 0) {
      f->sample_age_ns = s.age_ns;
      l->age_sum_ns += s.age_ns;
      l->age_max_ns = MAX2(l->age_max_ns, s.age_ns);
      l->age_n++;
   }

   if (use) {
      memcpy(f->eye, s.eye, sizeof(s.eye));
      memcpy(f->uv, s.uv, sizeof(s.uv));
      bool deg = false;
      for (unsigned e = 0; e < 2; e++)
         deg |= f->eye[e] && s.scale[e][0] > 0.f;
      /* Degrees only when every fovea eye has a published scale. */
      for (unsigned e = 0; e < 2; e++)
         if (f->eye[e] && !(s.scale[e][0] > 0.f))
            deg = false;
      if (deg)
         memcpy(f->scale, s.scale, sizeof(s.scale));
      unsigned li = tu_vrs_layout_index(c->layout);
      if (deg) {
         f->r0 = c->r0_deg > 0.f ? c->r0_deg : tu_vrs_layout_deg[li][0];
         f->r1 = c->r1_deg > 0.f ? c->r1_deg : tu_vrs_layout_deg[li][1];
      } else {
         f->r0 = c->r0_eh > 0.f ? c->r0_eh : tu_vrs_layout_eh[li][0];
         f->r1 = c->r1_eh > 0.f ? c->r1_eh : tu_vrs_layout_eh[li][1];
      }
      if (f->r1 < f->r0)
         f->r1 = f->r0;
      f->cap = c->max >= 4 ? 2 : c->max >= 2 ? 1 : 0;
      f->base_on = c->on;
      f->t_on = t_on;
      if (t_on) {
         float tr0 = deg ? c->t_r0_deg : c->t_r0_eh;
         float tr1 = deg ? c->t_r1_deg : c->t_r1_eh;
         f->t_r0 = tr0 > 0.f ? tr0 : f->r0;
         f->t_r1 = tr1 > 0.f ? tr1 : f->r1;
         if (f->t_r1 < f->t_r0)
            f->t_r1 = f->t_r0;
         f->t_cap = c->t_max >= 4 ? 2 : c->t_max >= 2 ? 1 : 0;
      }
      if (c->harsh) {
         f->r0 = deg ? 5.0f : 0.06f;
         f->r1 = f->r0;
         f->cap = 2;
         f->dither = 0;
         f->t_r0 = f->t_r1 = f->r0;
         f->t_cap = 2;
      } else if (f->flags & TU_GAZE_FLAG_SACCADE) {
         f->r0 *= vrs_cfg.saccade_scale;
         f->r1 *= vrs_cfg.saccade_scale;
         f->t_r0 *= vrs_cfg.saccade_scale;
         f->t_r1 *= vrs_cfg.saccade_scale;
      }
      /* Ramp in after (re)activation: 2x2 cap for the first half. */
      if (frame - l->ramp_start < (uint64_t) DIV_ROUND_UP(vrs_cfg.ramp, 2)) {
         f->cap = MIN2(f->cap, 1u);
         f->t_cap = MIN2(f->t_cap, 1u);
      }
      l->epochs_active++;
   } else {
      f->state = TU_VRS_OFF;
   }

   tu_vrs_log_transition(l);
   tu_vrs_log_interval(l);
}

void
tu_vrs_frame_get(struct tu_device *dev, struct tu_vrs_frame *f)
{
   simple_mtx_lock(&latch_mtx);
   uint64_t frame = tu_foveate_frame(dev);
   if (vrs_latch.key != frame) {
      vrs_latch.key = frame;
      tu_vrs_update(&vrs_latch, frame);
   }
   *f = vrs_latch.frame;
   simple_mtx_unlock(&latch_mtx);
}

static const uint8_t tu_vrs_codes[3] = { 0, 5, 10 };

void
tu_vrs_build_map(const struct tu_vrs_frame *f, uint32_t w, uint32_t h,
                 uint8_t *map, uint32_t pitch)
{
   uint32_t tw = DIV_ROUND_UP(w, TU_VRS_TEXEL), th = DIV_ROUND_UP(h, TU_VRS_TEXEL);
   uint32_t hist[3] = { 0, 0, 0 };
   if (f->state == TU_VRS_OFF) {
      for (uint32_t ty = 0; ty < th; ty++)
         memset(map + ty * pitch, 0, tw);
      hist[0] = tw * th;
   } else {
      bool sbs = (float) w >= vrs_cfg.sbs_min * (float) h;
      float ew = sbs ? w * 0.5f : (float) w, eh = (float) h;
      bool deg = f->scale[0][0] > 0.f || f->scale[1][0] > 0.f;
      for (uint32_t ty = 0; ty < th; ty++) {
         float py = ty * TU_VRS_TEXEL + TU_VRS_TEXEL * 0.5f;
         for (uint32_t tx = 0; tx < tw; tx++) {
            float px = tx * TU_VRS_TEXEL + TU_VRS_TEXEL * 0.5f;
            unsigned e0 = 0, e1 = 1;
            if (sbs)
               e0 = e1 = px >= ew;
            float d = INFINITY, unit = INFINITY;
            for (unsigned e = e0; e <= e1; e++) {
               if (!f->eye[e])
                  continue;
               float ex0 = sbs ? e * ew : 0.f;
               float du = (px - ex0) / ew - f->uv[e][0];
               float dv = py / eh - f->uv[e][1];
               float dd, uu;
               if (deg) {
                  float sx = f->scale[e][0], sy = f->scale[e][1];
                  dd = sqrtf(du * du * sx * sx + dv * dv * sy * sy);
                  uu = MAX2(sx / ew, sy / eh) * TU_VRS_TEXEL;
               } else {
                  dd = sqrtf(du * du * ew * ew + dv * dv * eh * eh) / eh;
                  uu = TU_VRS_TEXEL / eh;
               }
               if (dd < d) {
                  d = dd;
                  unit = uu;
               }
            }
            float band = f->dither * unit * 0.5f;
            bool chk = (tx + ty) & 1;
            unsigned lvl;
            if (d < f->r0 - band)
               lvl = 0;
            else if (d < f->r0 + band)
               lvl = chk ? 1 : 0;
            else if (d < f->r1 - band)
               lvl = 1;
            else if (d < f->r1 + band)
               lvl = chk ? 2 : 1;
            else
               lvl = 2;
            lvl = MIN2(lvl, f->cap);
            map[ty * pitch + tx] = tu_vrs_codes[lvl];
            hist[lvl]++;
         }
      }
   }
   p_atomic_inc(&vrs_stat_maps);
   for (unsigned i = 0; i < 3; i++)
      p_atomic_add(&vrs_stat_texels[i], hist[i]);
}

void
tu_vrs_dump_map(const struct tu_vrs_frame *f, uint32_t w, uint32_t h,
                const uint8_t *map, uint32_t pitch)
{
   if (!vrs_cfg.dump)
      return;
   uint32_t n = p_atomic_inc_return(&vrs_dumped);
   if (n > vrs_cfg.dump_max)
      return;
   char path[512];
   snprintf(path, sizeof(path), "%s.%u.%" PRIu64 ".%ux%u.pgm", vrs_cfg.dump, n,
            f->epoch, w, h);
   FILE *fp = fopen(path, "wb");
   if (!fp)
      return;
   uint32_t tw = DIV_ROUND_UP(w, TU_VRS_TEXEL), th = DIV_ROUND_UP(h, TU_VRS_TEXEL);
   fprintf(fp, "P5\n# tu-vrs epoch=%" PRIu64 " state=%u gaze=%.4f,%.4f/%.4f,%.4f eyes=%d%d r0=%.4f r1=%.4f cap=%u dither=%u\n%u %u\n255\n",
           f->epoch, f->state, f->uv[0][0], f->uv[0][1], f->uv[1][0], f->uv[1][1],
           f->eye[0], f->eye[1], f->r0, f->r1, f->cap, f->dither, tw, th);
   for (uint32_t ty = 0; ty < th; ty++)
      fwrite(map + ty * pitch, 1, tw, fp);
   fclose(fp);
   mesa_logi("tu-vrs: dumped rate map %s", path);
}

void
tu_vrs_note_pass(enum tu_vrs_pass_kind kind)
{
   p_atomic_inc(&vrs_stat_pass[kind]);
}

/* SPIR-V header scan for vkd3d-proton's module name, an OpString holding
 * exactly "<16 hex>.dxil" (or ".dxbc"). Stops at the first OpFunction.
 */
uint64_t
tu_spirv_dxil_hash(const uint32_t *words, size_t size)
{
   size_t n = size / sizeof(uint32_t);
   if (!words || n < 5 || words[0] != 0x07230203)
      return 0;
   for (size_t i = 5; i < n;) {
      uint32_t op = words[i] & 0xffff, len = words[i] >> 16;
      if (!len || len > n - i || op == 54 /* OpFunction */)
         break;
      if (op == 7 /* OpString */ && len > 2) {
         const char *str = (const char *) (words + i + 2);
         size_t sl = strnlen(str, (len - 2) * sizeof(uint32_t));
         if (sl == 21 && (!strcmp(str + 16, ".dxil") || !strcmp(str + 16, ".dxbc"))) {
            uint64_t v = 0;
            unsigned k;
            for (k = 0; k < 16; k++) {
               char c = str[k];
               unsigned d;
               if (c >= '0' && c <= '9')
                  d = c - '0';
               else if (c >= 'a' && c <= 'f')
                  d = c - 'a' + 10;
               else if (c >= 'A' && c <= 'F')
                  d = c - 'A' + 10;
               else
                  break;
               v = (v << 4) | d;
            }
            if (k == 16)
               return v;
         }
      }
      i += len;
   }
   return 0;
}

uint64_t
tu_stage_dxil_hash(const VkPipelineShaderStageCreateInfo *stage)
{
   if (!stage)
      return 0;
   VK_FROM_HANDLE(vk_shader_module, module, stage->module);
   if (module)
      return tu_spirv_dxil_hash((const uint32_t *) module->data, module->size);
   const VkShaderModuleCreateInfo *mci =
      vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO);
   if (mci)
      return tu_spirv_dxil_hash(mci->pCode, mci->codeSize);
   return 0;
}

void
tu_vrs_note_depthless_pass(void)
{
   p_atomic_inc(&vrs_stat_ps[VRS_PS_DEPTHLESS_PASS]);
}

/* tu_shader::vrs_ps_class encoding (0 = not classified yet). */
#define VRS_PS_CLS_UNLISTED 1u
#define VRS_PS_CLS_ALLOW    2u
#define VRS_PS_CLS_DENY     3u
#define VRS_PS_CLS_MASK     3u
#define VRS_PS_CLS_CAP_SHIFT 2  /* allow: coarsest level 0..2 */
#define VRS_PS_CLS_UNHASHED (1u << 4)

static uint8_t
tu_vrs_ps_classify(struct tu_shader *fs)
{
   uint8_t c = fs->vrs_ps_class;
   if (c)
      return c;
   uint64_t h = fs->fs.dxil_hash;
   uint8_t cap = 2;
   if (tu_vrs_ps_denied(h))
      c = VRS_PS_CLS_DENY;
   else if (h && tu_vrs_ps_match(&vrs_cfg.ps_allow, h, &cap))
      c = VRS_PS_CLS_ALLOW | (cap << VRS_PS_CLS_CAP_SHIFT);
   else
      c = VRS_PS_CLS_UNLISTED;
   if (!h)
      c |= VRS_PS_CLS_UNHASHED;
   fs->vrs_ps_class = c;
   if (vrs_cfg.ps_log &&
       ((c & VRS_PS_CLS_MASK) != VRS_PS_CLS_UNLISTED || vrs_cfg.ps_log >= 2)) {
      static const char *const names[4] = { "?", "unlisted", "allow", "deny" };
      unsigned r = 1u << ((c >> VRS_PS_CLS_CAP_SHIFT) & 3);
      mesa_logi("tu-vrs-ps: fs hash=%016" PRIx64 " class=%s%s cap=%ux%u bin=%.12s",
                h, names[c & VRS_PS_CLS_MASK],
                (c & VRS_PS_CLS_UNHASHED) ? "(unhashed)" : "",
                (c & VRS_PS_CLS_MASK) == VRS_PS_CLS_ALLOW ? r : 1u,
                (c & VRS_PS_CLS_MASK) == VRS_PS_CLS_ALLOW ? r : 1u,
                fs->variant ? fs->variant->blake3_str : "?");
   }
   return c;
}

bool
tu_vrs_pipeline_fsr(struct tu_shader *fs, uint32_t samples,
                    bool feedback_loop, enum tu_vrs_pass_scope scope,
                    struct vk_fragment_shading_rate_state *fsr)
{
   const struct ir3_shader_variant *v = fs ? fs->variant : NULL;
   /* A list failed to parse: fail closed. */
   if (vrs_cfg.ps_error) {
      p_atomic_inc(&vrs_stat_state[2]);
      return false;
   }
   uint8_t cap = 2;
   if (fs && v) {
      uint8_t c = tu_vrs_ps_classify(fs);
      if ((c & VRS_PS_CLS_MASK) == VRS_PS_CLS_DENY) {
         p_atomic_inc(&vrs_stat_ps[VRS_PS_DENY]);
         return false;
      }
      /* Allow mode: depth-less passes and inherited secondaries always need
       * an allowlisted FS; with TU_VRS_PS_MODE=strict (default) so do depth
       * passes.
       */
      if (vrs_cfg.ps_allow.set &&
          (scope != TU_VRS_SCOPE_DEPTH || vrs_cfg.ps_strict)) {
         if ((c & VRS_PS_CLS_MASK) != VRS_PS_CLS_ALLOW) {
            p_atomic_inc(&vrs_stat_ps[(c & VRS_PS_CLS_UNHASHED) ?
                                      VRS_PS_UNHASHED : VRS_PS_UNLISTED]);
            return false;
         }
         cap = (c >> VRS_PS_CLS_CAP_SHIFT) & 3;
         p_atomic_inc(&vrs_stat_ps[VRS_PS_ALLOW]);
         if (cap == 0) {
            p_atomic_inc(&vrs_stat_ps[VRS_PS_CAPPED]);
            return false;
         }
      }
   } else if (vrs_cfg.ps_allow.set) {
      p_atomic_inc(&vrs_stat_ps[VRS_PS_UNHASHED]);
      return false;
   }
   /* Side effects, depth/stencil/sample-mask export, sample-rate shading,
    * framebuffer/input-attachment reads and feedback loops: never coarse.
    */
   if (!v || v->no_earlyz || v->has_ssbo || v->writes_pos ||
       v->writes_stencilref || v->writes_smask || v->sample_shading ||
       fs->fs.sample_shading || v->fb_read ||
       fs->fs.dynamic_input_attachments_used || feedback_loop ||
       samples > 4) {
      p_atomic_inc(&vrs_stat_state[2]);
      return false;
   }
   /* The app's own attachment combiner must be trivial (KEEP), or with
    * TU_VRS_APP_COMB=max an identity op against the absent (1x1) app
    * attachment: MAX or MUL.
    */
   VkFragmentShadingRateCombinerOpKHR op1 = fsr->combiner_ops[1];
   if (op1 != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR) {
      if ((unsigned) op1 < 5)
         p_atomic_inc(&vrs_stat_appop[op1]);
      if (!vrs_cfg.app_comb_max ||
          (op1 != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR &&
           op1 != VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_KHR)) {
         p_atomic_inc(&vrs_stat_state[3]);
         return false;
      }
      p_atomic_inc(&vrs_stat_appop_accepted);
   }
   if (v->has_kill || cap == 1) {
      /* An app rate already >= 2x2 is at or above the cap: keep it, since
       * MIN with the map would shade finer than the app asked inside the
       * fovea.
       */
      if (fsr->fragment_size.width >= 2 && fsr->fragment_size.height >= 2) {
         if (v->has_kill)
            p_atomic_inc(&vrs_stat_state[1]);
         else
            p_atomic_inc(&vrs_stat_ps[VRS_PS_CAPPED]);
         return false;
      }
      /* Alpha-tested/masked, or an allowlist entry capped at 2x2: cap at 2x2
       * (MIN with a >=2x2 pipeline rate).
       */
      fsr->fragment_size.width = MAX2(fsr->fragment_size.width, 2);
      fsr->fragment_size.height = MAX2(fsr->fragment_size.height, 2);
      fsr->combiner_ops[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR;
      if (v->has_kill)
         p_atomic_inc(&vrs_stat_state[1]);
      else
         p_atomic_inc(&vrs_stat_ps[VRS_PS_CAPPED]);
   } else {
      /* App/material coarse rates still win. */
      fsr->combiner_ops[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR;
      p_atomic_inc(&vrs_stat_state[0]);
   }
   return true;
}
