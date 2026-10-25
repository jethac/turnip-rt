/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_d1.h"

#include <inttypes.h>

#include "util/log.h"
#include "util/os_misc.h"
#include "util/set.h"
#include "util/u_atomic.h"
#include "vk_format.h"
#include "vk_util.h"

#include "fdl/fd6_format_table.h"

#include "tu_cmd_buffer.h"
#include "tu_cs.h"
#include "tu_device.h"
#include "tu_image.h"

struct tu_d1_sig {
   uint32_t buf_info;
   uint32_t pitch;
   uint32_t array_pitch;
   uint32_t sp_ps_mrt;
   uint32_t flag_pitch;
   uint32_t ubwc;
   uint32_t ubwc_layer_size;
   uint32_t width;
   uint32_t height;
   uint32_t layers;
   uint32_t samples;
   uint32_t format;
};

struct tu_d1_device {
   uint32_t mode;
   struct tu_bo *bo;
   mtx_t mtx;
   struct set *views;
   uint32_t nsig;
   struct tu_d1_sig sigs[TU_D1_MAX_SIGS];

   uint32_t pub_ok;
   uint32_t pub_rej;
   uint32_t use;
   uint32_t use_bad;
   uint32_t unknown;
   uint32_t events;

   uint32_t warned_unknown;
   uint32_t warned_pub_rej;
   uint32_t warned_use_bad;
   uint32_t warned_full;
};

uint32_t
tu_d1_physical_device_mode(const struct tu_physical_device *pdevice)
{
   const char *s = os_get_option("TU_D1_FORWARD");
   if (!s || !*s || !strcmp(s, "0"))
      return 0;

   uint32_t mode;
   if (!strcmp(s, "1")) {
      mode = 1;
   } else if (!strcmp(s, "log")) {
      mode = 2;
   } else {
      mesa_logw("tu d1: TU_D1_FORWARD=%s not understood, forward off", s);
      return 0;
   }

   if (pdevice->info->chip != A7XX) {
      mesa_logw("tu d1: TU_D1_FORWARD=%s needs an a7xx GPU, forward off", s);
      return 0;
   }

   return mode;
}

void
tu_d1_driver_info(const struct tu_physical_device *pdevice, char *info,
                  size_t size)
{
   if (!pdevice->d1_mode)
      return;
   size_t n = strnlen(info, size);
   if (n < size)
      snprintf(info + n, size - n, " d1fwd=1");
}

static void
tu_d1_log_stats(struct tu_d1_device *d1)
{
   mesa_logi("tu d1: pub_ok=%u pub_rej=%u use=%u use_bad=%u unknown=%u",
             p_atomic_read(&d1->pub_ok), p_atomic_read(&d1->pub_rej),
             p_atomic_read(&d1->use), p_atomic_read(&d1->use_bad),
             p_atomic_read(&d1->unknown));
}

static bool
tu_d1_once(uint32_t *flag)
{
   return p_atomic_cmpxchg(flag, 0, 1) == 0;
}

void
tu_d1_device_init(struct tu_device *dev)
{
   dev->d1 = NULL;

   const uint32_t mode = dev->physical_device->d1_mode;
   if (!mode)
      return;

   struct tu_d1_device *d1 = (struct tu_d1_device *)
      vk_zalloc(&dev->vk.alloc, sizeof(*d1), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!d1) {
      mesa_loge("tu d1: out of memory, forward off");
      return;
   }

   d1->mode = mode;
   d1->views = _mesa_pointer_set_create(NULL);
   if (!d1->views) {
      vk_free(&dev->vk.alloc, d1);
      mesa_loge("tu d1: out of memory, forward off");
      return;
   }

   const uint64_t size =
      (uint64_t) TU_D1_SLOTS * TU_D1_MAX_SIGS * TU_D1_REC_DWORDS * 4;
   VkResult result = tu_bo_init_new(
      dev, NULL, &d1->bo, size,
      (enum tu_bo_alloc_flags) (TU_BO_ALLOC_ALLOW_DUMP |
                                TU_BO_ALLOC_INTERNAL_RESOURCE),
      "d1_slots");
   if (result == VK_SUCCESS) {
      result = tu_bo_map(dev, d1->bo, NULL);
      if (result != VK_SUCCESS)
         tu_bo_finish(dev, d1->bo);
   }
   if (result != VK_SUCCESS) {
      _mesa_set_destroy(d1->views, NULL);
      vk_free(&dev->vk.alloc, d1);
      mesa_loge("tu d1: slot BO allocation failed, forward off");
      return;
   }
   memset(d1->bo->map, 0, size);

   mtx_init(&d1->mtx, mtx_plain);
   dev->d1 = d1;

   mesa_logi("tu d1: TU_D1_FORWARD=%s slots=%u sigs=%u bo=0x%" PRIx64,
             mode == 2 ? "log" : "1", TU_D1_SLOTS, TU_D1_MAX_SIGS,
             d1->bo->iova);
}

void
tu_d1_device_finish(struct tu_device *dev)
{
   struct tu_d1_device *d1 = dev->d1;
   if (!d1)
      return;

   if (d1->mode == 2)
      tu_d1_log_stats(d1);

   tu_bo_finish(dev, d1->bo);
   _mesa_set_destroy(d1->views, NULL);
   mtx_destroy(&d1->mtx);
   vk_free(&dev->vk.alloc, d1);
   dev->d1 = NULL;
}

void
tu_d1_view_created(struct tu_device *dev, struct tu_image_view *iview)
{
   struct tu_d1_device *d1 = dev->d1;
   mtx_lock(&d1->mtx);
   _mesa_set_add(d1->views, iview);
   mtx_unlock(&d1->mtx);
}

void
tu_d1_view_destroyed(struct tu_device *dev, struct tu_image_view *iview)
{
   struct tu_d1_device *d1 = dev->d1;
   mtx_lock(&d1->mtx);
   _mesa_set_remove_key(d1->views, iview);
   mtx_unlock(&d1->mtx);
}

static const char *
tu_d1_check_view(const struct tu_device *dev,
                 const struct tu_image_view *iview)
{
   if (iview->vk.base.type != VK_OBJECT_TYPE_IMAGE_VIEW ||
       iview->vk.base.device != &dev->vk)
      return "not an image view of this device";

   const struct tu_image *image = iview->image;
   if (!image || !image->iova)
      return "image not bound";
   if (iview->vk.view_type != VK_IMAGE_VIEW_TYPE_2D &&
       iview->vk.view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY)
      return "not a 2D view";
   if (iview->vk.level_count != 1)
      return "not a single-level view";
   if (image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
      return "not single-sample";
   if (iview->vk.aspects != VK_IMAGE_ASPECT_COLOR_BIT ||
       vk_format_get_plane_count(iview->vk.format) != 1)
      return "not a single-plane colour view";
   if (!(iview->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      return "no colour attachment usage";
   if (image->vk.create_flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                 VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT))
      return "sparse or subsampled image";
   if (fd6_color_format(vk_format_to_pipe_format(iview->vk.format),
                        (enum a6xx_tile_mode) image->layout[0].tile_mode) ==
       FMT6_NONE)
      return "format not renderable";
   return NULL;
}

static void
tu_d1_view_sig(const struct tu_image_view *iview,
               const struct tu_mrt_values *v, struct tu_d1_sig *sig)
{
   memset(sig, 0, sizeof(*sig));
   sig->buf_info = v->buf_info;
   sig->pitch = v->pitch;
   sig->array_pitch = v->array_pitch;
   sig->sp_ps_mrt = v->sp_ps_mrt;
   sig->flag_pitch = v->flag_pitch;
   sig->ubwc = iview->view.ubwc_enabled;
   sig->ubwc_layer_size = iview->view.ubwc_layer_size;
   sig->width = iview->view.width;
   sig->height = iview->view.height;
   sig->layers = iview->vk.layer_count;
   sig->samples = iview->image->vk.samples;
   sig->format = iview->vk.format;
}

static uint32_t
tu_d1_sig_id_locked(struct tu_d1_device *d1, const struct tu_d1_sig *sig)
{
   for (uint32_t i = 0; i < d1->nsig; i++) {
      if (!memcmp(&d1->sigs[i], sig, sizeof(*sig)))
         return i + 1;
   }
   if (d1->nsig == TU_D1_MAX_SIGS)
      return 0;
   d1->sigs[d1->nsig++] = *sig;
   return d1->nsig;
}

static const char *
tu_d1_lookup(struct tu_device *dev, const struct tu_image_view *iview,
             bool registered_only, struct tu_mrt_values *v,
             uint32_t *sig_id)
{
   struct tu_d1_device *d1 = dev->d1;
   const char *why = NULL;

   mtx_lock(&d1->mtx);
   if (registered_only && !_mesa_set_search(d1->views, iview)) {
      why = "not a live image view";
   } else {
      why = tu_d1_check_view(dev, iview);
      if (!why) {
         struct tu_d1_sig sig;
         *v = tu_image_view_mrt_values(iview);
         tu_d1_view_sig(iview, v, &sig);
         *sig_id = tu_d1_sig_id_locked(d1, &sig);
         if (!*sig_id)
            why = "signature table full";
      }
   }
   mtx_unlock(&d1->mtx);

   if (why && !strcmp(why, "signature table full") &&
       tu_d1_once(&d1->warned_full))
      mesa_logw("tu d1: more than %u attachment layouts, forward off for new ones",
                TU_D1_MAX_SIGS);
   return why;
}

static uint64_t
tu_d1_rec_iova(const struct tu_d1_device *d1, uint32_t slot, uint32_t sig_id)
{
   return d1->bo->iova +
          (uint64_t) tu_d1_rec_index(slot, sig_id) * TU_D1_REC_DWORDS * 4;
}

static void
tu_d1_publish(struct tu_cmd_buffer *cmd, const struct tu_d1_label *l)
{
   struct tu_d1_device *d1 = cmd->device->d1;
   const char *why = NULL;
   struct tu_mrt_values v;
   uint32_t sig_id = 0;

   if (cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
       (cmd->usage_flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT))
      why = "inside a render pass continue secondary";
   else if (!l->view)
      why = "null view";
   else
      why = tu_d1_lookup(cmd->device,
                         (const struct tu_image_view *) (uintptr_t) l->view,
                         true, &v, &sig_id);

   if (why) {
      p_atomic_inc(&d1->pub_rej);
      if (tu_d1_once(&d1->warned_pub_rej))
         mesa_logw("tu d1: D1PUB s=%u g=%u v=%016" PRIx64 " rejected: %s",
                   l->slot, l->gen, l->view, why);
      return;
   }

   uint32_t rec[TU_D1_REC_WRITTEN];
   rec[TU_D1_REC_BUF_INFO] = v.buf_info;
   rec[TU_D1_REC_PITCH] = (uint32_t) A6XX_RB_MRT_PITCH(0, v.pitch).value;
   rec[TU_D1_REC_ARRAY_PITCH] =
      (uint32_t) A6XX_RB_MRT_ARRAY_PITCH(0, v.array_pitch).value;
   rec[TU_D1_REC_BASE_LO] = (uint32_t) v.base;
   rec[TU_D1_REC_BASE_HI] = (uint32_t) (v.base >> 32);
   rec[TU_D1_REC_SP_PS_MRT] = v.sp_ps_mrt;
   rec[TU_D1_REC_FLAG_LO] = (uint32_t) v.flag;
   rec[TU_D1_REC_FLAG_HI] = (uint32_t) (v.flag >> 32);
   rec[TU_D1_REC_FLAG_PITCH] = v.flag_pitch;
   rec[TU_D1_REC_TAG] = (l->slot << 16) | sig_id;
   rec[TU_D1_REC_KEY] = l->gen;

   struct tu_cs *cs = &cmd->cs;
   tu_cs_reserve(cs, 3 + 3 + TU_D1_REC_WRITTEN);
   tu_cond_exec_start(cs, CP_COND_REG_EXEC_0_MODE(THREAD_MODE) |
                          CP_COND_REG_EXEC_0_BR);
   tu_cs_emit_pkt7(cs, CP_MEM_WRITE, 2 + TU_D1_REC_WRITTEN);
   tu_cs_emit_qw(cs, tu_d1_rec_iova(d1, l->slot, sig_id));
   tu_cs_emit_array(cs, rec, TU_D1_REC_WRITTEN);
   tu_cond_exec_end(cs);

   p_atomic_inc(&d1->pub_ok);
}

static void
tu_d1_use_bad(struct tu_d1_device *d1, uint32_t gen, const char *why)
{
   p_atomic_inc(&d1->use_bad);
   if (tu_d1_once(&d1->warned_use_bad))
      mesa_logw("tu d1: D1USE g=%u not applied: %s", gen, why);
}

void
tu_d1_disarm(struct tu_cmd_buffer *cmd, const char *why)
{
   struct tu_d1_cmd *st = &cmd->state.d1;
   if (!st->armed)
      return;
   st->armed = false;
   tu_d1_use_bad(cmd->device->d1, st->gen, why);
}

void
tu_d1_label(struct tu_cmd_buffer *cmd, const char *name)
{
   struct tu_d1_device *d1 = cmd->device->d1;
   struct tu_d1_cmd *st = &cmd->state.d1;
   struct tu_d1_label l;

   switch (tu_d1_parse_label(name, &l)) {
   case TU_D1_LABEL_PUB:
      tu_d1_publish(cmd, &l);
      break;
   case TU_D1_LABEL_USE:
      tu_d1_disarm(cmd, "superseded by a later D1USE");
      st->armed = true;
      st->gen = l.gen;
      st->slot[0] = l.a6;
      st->slot[1] = l.a7;
      break;
   default:
      p_atomic_inc(&d1->unknown);
      if (tu_d1_once(&d1->warned_unknown))
         mesa_logw("tu d1: ignoring malformed label \"%.64s\"", name);
      break;
   }

   if (d1->mode == 2 && p_atomic_inc_return(&d1->events) % 1000 == 0)
      tu_d1_log_stats(d1);
}

void
tu_d1_begin_rendering(struct tu_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct tu_d1_device *d1 = cmd->device->d1;
   struct tu_d1_cmd *st = &cmd->state.d1;
   const char *why = NULL;

   st->armed = false;
   memset(st->att, 0, sizeof(st->att));

   const VkMultisampledRenderToSingleSampledInfoEXT *msrtss =
      vk_find_struct_const(info->pNext,
                           MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT);

   if (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT))
      why = "suspending or resuming render pass";
   else if (info->viewMask)
      why = "multiview render pass";
   else if (msrtss && msrtss->multisampledRenderToSingleSampledEnable)
      why = "multisampled render to single sampled";
   else if (info->colorAttachmentCount < 8)
      why = "fewer than 8 colour attachments";

   for (unsigned j = 0; j < 2 && !why; j++) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[6 + j];
      struct tu_mrt_values v;
      uint32_t sig_id = 0;

      if (!att->imageView)
         why = "null colour attachment 6 or 7";
      else if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         why = "colour attachment 6 or 7 is cleared on load";
      else if (att->resolveMode != VK_RESOLVE_MODE_NONE)
         why = "colour attachment 6 or 7 resolves";
      else
         why = tu_d1_lookup(cmd->device,
                            tu_image_view_from_handle(att->imageView), false,
                            &v, &sig_id);
      if (why)
         break;

      st->att[j].valid = true;
      st->att[j].key = st->gen;
      st->att[j].rec_iova = tu_d1_rec_iova(d1, st->slot[j], sig_id);
   }

   if (why) {
      memset(st->att, 0, sizeof(st->att));
      tu_d1_use_bad(d1, st->gen, why);
      return;
   }

   st->rp_active = true;
   cmd->state.rp.disable_gmem = true;
   cmd->state.rp.force_render_mode_reason = "TU_D1_FORWARD";
   p_atomic_inc(&d1->use);
}

static struct fd_reg_pair
tu_d1_reg(uint32_t reg)
{
   struct fd_reg_pair r = {};
   r.reg = reg;
   return r;
}

template <chip CHIP>
void
tu_d1_emit_forward(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   if (CHIP != A7XX)
      return;

   const struct tu_d1_cmd *st = &cmd->state.d1;
   const struct tu_subpass *subpass = cmd->state.subpass;
   unsigned mrt[2], idx[2], n = 0;

   for (unsigned j = 0; j < 2; j++) {
      const unsigned i = 6 + j;
      if (!st->att[j].valid || i >= subpass->color_count ||
          subpass->color_attachments[i].attachment == VK_ATTACHMENT_UNUSED)
         continue;
      const unsigned m = cmd->vk.dynamic_graphics_state.cal.color_map[i];
      if (m == MESA_VK_ATTACHMENT_UNUSED || m >= MAX_RTS)
         continue;
      mrt[n] = m;
      idx[n] = j;
      n++;
   }
   if (!n)
      return;

   const struct fd_reg_pair scratch = tu_scratch_reg<CHIP>(TU_D1_SCRATCH);

   tu_cs_reserve(cs, 3 + 2 + n * (4 + 1 + 4 + 4 + 4));
   tu_cond_exec_start(cs, CP_COND_EXEC_0_RENDER_MODE_SYSMEM);
   tu_cs_emit_pkt7(cs, CP_WAIT_MEM_WRITES, 0);
   tu_cs_emit_pkt7(cs, CP_WAIT_FOR_ME, 0);
   for (unsigned k = 0; k < n; k++) {
      const uint64_t rec = st->att[idx[k]].rec_iova;
      cs->mem_to_reg(scratch, rec + 4 * TU_D1_REC_KEY,
                     { .one_reg_wr = true, .wait_cache_flush = true });
      tu_cs_emit_pkt7(cs, CP_WAIT_FOR_ME, 0);
      tu_cond_exec_start_imm(cs,
         CP_COND_REG_EXEC_0_MODE(REG_COMPARE_IMM) |
         CP_COND_REG_EXEC_0_REG0(scratch.reg), st->att[idx[k]].key);
      cs->mem_to_reg(tu_d1_reg(REG_A6XX_RB_MRT_BASE(mrt[k])),
                     rec + 4 * TU_D1_REC_BASE_LO, { .cnt = 2 });
      cs->mem_to_reg(tu_d1_reg(REG_A6XX_RB_COLOR_FLAG_BUFFER_ADDR(mrt[k])),
                     rec + 4 * TU_D1_REC_FLAG_LO, { .cnt = 2 });
      tu_cond_exec_end(cs);
   }
   tu_cond_exec_end(cs);
}
TU_GENX(tu_d1_emit_forward);
