#include "rd_private.h"
#include "rd_sampling.h"
#include "compiler/nir/nir.h"
#include "vk_format.h"
#include "util/u_debug.h"

VkResult
rd_prepare_sampling(struct rd_device *device, struct rd_dispatch *dispatch)
{
   struct rd_pipeline *source = dispatch->pipeline;
   if (!source->sampling_nir)
      return VK_SUCCESS;
   struct rd_sampling_key key = {0};
   struct fd_bo *bos[16] = {0};
   uint64_t addresses[16] = {0};
   bool force = debug_get_bool_option("RADISH_SW_TEXEL_FETCH", false);
   for (unsigned i = 0; i < source->resource_usage.texture_count; i++) {
      if (!(source->resource_usage.textures & (1u << i)))
         continue;
      const struct rd_descriptor *descriptor = &dispatch->descriptors[source->texture_descriptors[i]];
      if (!descriptor->image_view)
         continue;
      const struct vk_image_view *view = &descriptor->image_view->vk;
      const struct rd_image *image = container_of(view->image, struct rd_image, vk);
      if (vk_format_is_compressed(image->vk.format))
         continue;
      bool array = view->view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      bool cube = view->view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
                  view->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      if (!array && !cube)
         continue;
      if (!force && image->levels[view->base_mip_level].array_pitch <= (0x3fffull << 12))
         continue;
      if (!image->memory || view->level_count > RD_IMAGE_MAX_LEVELS)
         return VK_ERROR_UNKNOWN;
      key.mask |= 1u << i;
      struct rd_sampling_view *target = &key.views[i];
      target->format = view->format;
      if (view->format == VK_FORMAT_D24_UNORM_S8_UINT)
         target->format = view->view_format;
      target->swizzle = view->swizzle;
      target->levels = view->level_count;
      target->layers = view->layer_count;
      target->cpp = image->texel_size;
      target->samples = image->vk.samples;
      for (unsigned l = 0; l < view->level_count; l++) {
         unsigned level = view->base_mip_level + l;
         const struct rd_image_level *layout = &image->levels[level];
         VkExtent3D extent = vk_image_mip_level_extent(&image->vk, level);
         VkDeviceSize offset = layout->offset + view->base_array_layer * layout->array_pitch;
         if (offset > UINT32_MAX || layout->row_pitch > UINT32_MAX || layout->array_pitch > UINT32_MAX)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         target->level[l] = (struct rd_sampling_level) {
            .offset = offset, .row_pitch = layout->row_pitch, .array_pitch = layout->array_pitch,
            .width = extent.width, .height = extent.height,
         };
      }
      bos[i] = image->memory->bo;
      addresses[i] = fd_bo_get_iova(bos[i]) + image->memory_offset;
   }
   if (!key.mask)
      return VK_SUCCESS;
   nir_foreach_function_impl(impl, source->sampling_nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->texture_index >= 16 || !(key.mask & (1u << tex->texture_index)) ||
                tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms ||
                tex->op == nir_texop_texture_samples || tex->op == nir_texop_txs ||
                tex->op == nir_texop_query_levels)
               continue;
            if (tex->sampler_index >= 16)
               return VK_ERROR_FEATURE_NOT_PRESENT;
            const struct rd_descriptor *sampler =
               &dispatch->descriptors[source->sampler_descriptors[tex->sampler_index]];
            if (!sampler->has_sampler)
               return VK_ERROR_UNKNOWN;
            memcpy(key.samplers[tex->sampler_index], sampler->sampler, sizeof(sampler->sampler));
         }
      }
   }
   struct rd_pipeline *pipeline;
   VkResult result = rd_sampling_compile(device, source, &key, &pipeline);
   if (result != VK_SUCCESS)
      return result;
   struct fd_bo *metadata = fd_bo_new(device->fd, 4096, FD_BO_GPUREADONLY, "radish-sampling-addresses");
   if (!metadata)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   void *map = fd_bo_map(metadata);
   if (!map) {
      fd_bo_del(metadata);
      return VK_ERROR_MEMORY_MAP_FAILED;
   }
   memcpy(map, addresses, sizeof(addresses));
   if (debug_get_bool_option("RADISH_RESOURCE_DIAGNOSTICS", false)) {
      const struct ir3_shader_variant *v = pipeline->variant;
      fprintf(stderr, "RADISH_SAMPLING_DIAG mask=%x native=%u full=%d half=%d branch=%u constlen=%u private=%u buffers=%u textures=%u samplers=%u aliases=%u metadata=%llx size=%u shader=%llx size=%u\n",
         key.mask, v->instrlen, v->info.max_reg + 1, v->info.max_half_reg + 1,
         ir3_shader_branchstack_hw(v), v->constlen, v->pvtmem_size, pipeline->buffer_count,
         pipeline->resource_usage.texture_count, pipeline->resource_usage.sampler_count,
         v->image_mapping.num_tex, (unsigned long long)fd_bo_get_iova(metadata), fd_bo_size(metadata),
         (unsigned long long)fd_bo_get_iova(pipeline->bo), fd_bo_size(pipeline->bo));
      for (unsigned i = 0; i < 16; i++) {
         if (!(key.mask & (1u << i))) continue;
         const struct rd_sampling_view *view = &key.views[i];
         fprintf(stderr, "RADISH_SAMPLING_BO texture=%u address=%llx bo_size=%u format=%u cpp=%u samples=%u layers=%u levels=%u offset=%u row=%u array=%u\n",
            i, (unsigned long long)addresses[i], fd_bo_size(bos[i]), view->format,
            view->cpp, view->samples, view->layers, view->levels, view->level[0].offset,
            view->level[0].row_pitch, view->level[0].array_pitch);
      }
      for (unsigned i = 0; i < source->buffer_count; i++) {
         const struct rd_descriptor *desc = &dispatch->descriptors[source->buffer_descriptors[i]];
         if (!desc->buffer || !desc->buffer->memory) continue;
         struct fd_bo *bo = desc->buffer->memory->bo;
         fprintf(stderr, "RADISH_SAMPLING_BUFFER index=%u address=%llx bo_size=%u range=%llu\n", i,
            (unsigned long long)(fd_bo_get_iova(bo) + desc->buffer->offset + desc->offset),
            fd_bo_size(bo), (unsigned long long)desc->range);
      }
      fflush(stderr);
   }
   dispatch->sampling_addresses = metadata;
   memcpy(dispatch->sampling_bos, bos, sizeof(bos));
   dispatch->pipeline = pipeline;
   return VK_SUCCESS;
}
