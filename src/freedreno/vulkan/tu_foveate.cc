/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_foveate.h"

#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

#include "util/u_debug.h"
#include "util/os_time.h"

#include "tu_cmd_buffer.h"
#include "tu_device.h"
#include "tu_image.h"
#include "tu_pass.h"

struct tu_foveate_cfg {
   bool enabled;
   float r0, r1;
   float fixed_x, fixed_y;
   uint32_t min_w, min_h;
   uint32_t max_area;
   uint64_t stale_ns;
   const char *path;
};

static struct tu_foveate_cfg cfg;
static const struct tu_gaze_shm *volatile shm;
static int64_t shm_retry_ns;

static float
env_f(const char *name, float def)
{
   const char *s = os_get_option(name);
   return s ? strtof(s, NULL) : def;
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
   cfg.stale_ns = (uint64_t) debug_get_num_option("TU_FOVEATE_STALE_MS", 250) * 1000000ull;
   cfg.path = os_get_option("TU_FOVEATE_SHM");
   if (!cfg.path)
      cfg.path = "/dev/shm/tu-gaze";
   if (cfg.enabled)
      mesa_logi("tu-foveate: r0=%.3f r1=%.3f min=%ux%u max=%u shm=%s",
                cfg.r0, cfg.r1, cfg.min_w, cfg.min_h, cfg.max_area, cfg.path);
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

static void
tu_foveate_gaze(unsigned eye, float *gx, float *gy)
{
   *gx = cfg.fixed_x;
   *gy = cfg.fixed_y;
   const struct tu_gaze_shm *p = tu_foveate_map();
   if (!p)
      return;
   uint32_t s0 = p->seq;
   __atomic_thread_fence(__ATOMIC_ACQUIRE);
   float x = p->uv[eye][0], y = p->uv[eye][1];
   uint32_t valid = p->valid;
   uint64_t t = p->time_ns;
   __atomic_thread_fence(__ATOMIC_ACQUIRE);
   if (s0 != p->seq || (s0 & 1) || !valid)
      return;
   if ((uint64_t) os_time_get_nano() - t > cfg.stale_ns)
      return;
   if (!(x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f))
      return;
   *gx = x;
   *gy = y;
}

bool
tu_foveate_pass_eligible(struct tu_cmd_buffer *cmd)
{
   if (!tu_foveate_enabled())
      return false;
   const struct tu_render_pass *pass = cmd->state.pass;
   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   if (!pass || !fb || !pass->has_fdm ||
       pass->fragment_density_map.attachment != VK_ATTACHMENT_UNUSED)
      return false;
   if (fb->width < cfg.min_w || fb->height < cfg.min_h)
      return false;
   for (uint32_t i = 0; i < pass->subpass_count; i++)
      if (pass->subpasses[i].color_count)
         return true;
   return false;
}

void
tu_foveate_frag_area(struct tu_cmd_buffer *cmd,
                     uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                     unsigned view, struct tu_frag_area *area)
{
   area->width = area->height = 1.0f;
   if (!tu_foveate_pass_eligible(cmd))
      return;

   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   uint32_t w = fb->width, h = fb->height;
   x2 = MIN2(x2, w);
   y2 = MIN2(y2, h);
   if (x1 >= x2 || y1 >= y2)
      return;

   bool sbs = w * 10 >= h * 16;
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
   tu_foveate_gaze(eye, &gx, &gy);
   float px = ex0 + gx * ew, py = gy * eh;
   float nx = CLAMP(px, (float) x1, (float) x2), ny = CLAMP(py, (float) y1, (float) y2);
   float d = sqrtf((nx - px) * (nx - px) + (ny - py) * (ny - py)) / eh;

   float a = d < cfg.r0 ? 1.f : d < cfg.r1 ? 2.f : 4.f;
   a = MIN2(a, (float) cfg.max_area);
   area->width = area->height = a;
}
