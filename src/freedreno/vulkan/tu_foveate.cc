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
