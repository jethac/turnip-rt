#include "rd_private.h"
#include "vk_util.h"
#include "vk_format.h"

static void
rd_command_reset(struct vk_command_buffer *base, VkCommandBufferResetFlags flags)
{
   struct rd_command_buffer *command = container_of(base, struct rd_command_buffer, vk);
   util_dynarray_foreach(&command->commands, struct rd_command, recorded) {
      if (recorded->type == RD_CMD_UPDATE)
         vk_free(&base->pool->alloc, recorded->transfer.data);
      else if (recorded->type == RD_CMD_RENDER)
         rd_render_free(command, recorded->render);
   }
   vk_command_buffer_reset(base);
   command->pipeline = NULL;
   command->gfx_pipeline = NULL;
   command->render = NULL;
   command->occlusion_pool = NULL;
   for (unsigned i = 0; i < 2; i++) {
      memset(command->bind[i].sets, 0, sizeof(command->bind[i].sets));
      memset(command->bind[i].dynamic_offsets, 0, sizeof(command->bind[i].dynamic_offsets));
   }
   memset(command->vertex_buffers, 0, sizeof(command->vertex_buffers));
   command->index_buffer = NULL;
   memset(command->push, 0, sizeof(command->push));
   util_dynarray_clear(&command->commands);
}

static void
rd_command_destroy(struct vk_command_buffer *base)
{
   struct rd_command_buffer *command = container_of(base, struct rd_command_buffer, vk);
   VkAllocationCallbacks alloc = base->pool->alloc;
   util_dynarray_foreach(&command->commands, struct rd_command, recorded) {
      if (recorded->type == RD_CMD_UPDATE)
         vk_free(&base->pool->alloc, recorded->transfer.data);
      else if (recorded->type == RD_CMD_RENDER)
         rd_render_free(command, recorded->render);
   }
   util_dynarray_fini(&command->commands);
   vk_command_buffer_finish(base);
   vk_free(&alloc, command);
}

static VkResult
rd_command_create(struct vk_command_pool *pool, VkCommandBufferLevel level,
                   struct vk_command_buffer **out)
{
   struct rd_command_buffer *command = vk_zalloc(&pool->alloc, sizeof(*command),
      8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!command)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = vk_command_buffer_init(pool, &command->vk,
      &rd_command_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, command);
      return result;
   }
   util_dynarray_init(&command->commands, NULL);
   *out = &command->vk;
   return VK_SUCCESS;
}

const struct vk_command_buffer_ops rd_command_buffer_ops = {
   .create = rd_command_create,
   .reset = rd_command_reset,
   .destroy = rd_command_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
rd_BeginCommandBuffer(VkCommandBuffer handle, const VkCommandBufferBeginInfo *info)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   const VkDeviceGroupCommandBufferBeginInfo *group =
      vk_find_struct_const(info->pNext, DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO);
   if (group && group->deviceMask != 1)
      return VK_ERROR_INITIALIZATION_FAILED;
   rd_command_reset(&command->vk, 0);
   vk_command_buffer_begin(&command->vk, info);
   command->render_continue = command->vk.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
      (info->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_EndCommandBuffer(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   return vk_command_buffer_end(&command->vk);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBindPipeline(VkCommandBuffer handle, VkPipelineBindPoint bind_point,
                   VkPipeline pipeline)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   struct rd_pipeline *bound = rd_pipeline_from_handle(pipeline);
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && bound->gfx) {
      command->gfx_pipeline = bound;
      vk_cmd_set_dynamic_graphics_state(&command->vk, &bound->gfx->dynamic);
      return;
   }
   if (bind_point != VK_PIPELINE_BIND_POINT_COMPUTE || bound->gfx) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   command->pipeline = bound;
}

static void
rd_layout_compatibility(const struct vk_pipeline_layout *layout,
                         uint8_t hashes[RD_MAX_SETS][BLAKE3_OUT_LEN])
{
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   VkPushConstantRange ranges[MESA_VK_MAX_PUSH_CONSTANT_RANGES];
   memcpy(ranges, layout->push_ranges, layout->push_range_count * sizeof(*ranges));
   for (uint32_t i = 1; i < layout->push_range_count; i++) {
      VkPushConstantRange range = ranges[i];
      uint32_t j = i;
      while (j && (ranges[j - 1].stageFlags > range.stageFlags ||
             (ranges[j - 1].stageFlags == range.stageFlags && ranges[j - 1].offset > range.offset) ||
             (ranges[j - 1].stageFlags == range.stageFlags && ranges[j - 1].offset == range.offset &&
              ranges[j - 1].size > range.size))) {
         ranges[j] = ranges[j - 1];
         j--;
      }
      ranges[j] = range;
   }
   _mesa_blake3_update(&hash, &layout->push_range_count, sizeof(layout->push_range_count));
   _mesa_blake3_update(&hash, ranges, layout->push_range_count * sizeof(*ranges));
   memset(hashes, 0, RD_MAX_SETS * BLAKE3_OUT_LEN);
   for (uint32_t i = 0; i < MIN2(layout->set_count, RD_MAX_SETS); i++) {
      _mesa_blake3_update(&hash, layout->set_layouts[i]->blake3, BLAKE3_OUT_LEN);
      _mesa_blake3_final(&hash, hashes[i]);
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBindDescriptorSets(VkCommandBuffer handle, VkPipelineBindPoint bind_point,
   VkPipelineLayout layout, uint32_t first, uint32_t count,
   const VkDescriptorSet *sets, uint32_t dynamic_count, const uint32_t *offsets)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if ((bind_point != VK_PIPELINE_BIND_POINT_COMPUTE &&
        bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) ||
       first > RD_MAX_SETS || count > RD_MAX_SETS - first) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, layout);
   struct rd_bind_state *bound = &command->bind[bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS];
   uint8_t compatibility[RD_MAX_SETS][BLAKE3_OUT_LEN];
   rd_layout_compatibility(pipeline_layout, compatibility);
   if (count) {
      uint32_t last = first + count - 1;
      for (uint32_t i = 0; i < RD_MAX_SETS; i++) {
         if (!bound->sets[i] || (i >= first && i <= last))
            continue;
         uint32_t compare = i < first ? i : last;
         if (memcmp(bound->set_compatibility[i][compare], compatibility[compare], BLAKE3_OUT_LEN))
            bound->sets[i] = NULL;
      }
   }
   uint32_t dynamic = 0;
   for (uint32_t i = 0; i < count; i++) {
      struct rd_descriptor_set *set = rd_descriptor_set_from_handle(sets[i]);
      bound->sets[first + i] = set;
      memcpy(bound->set_compatibility[first + i], compatibility, sizeof(compatibility));
      memset(bound->dynamic_offsets[first + i], 0,
             sizeof(bound->dynamic_offsets[first + i]));
      for (uint32_t j = 0; j < set->layout->binding_count; j++) {
         const struct rd_binding *binding = &set->layout->bindings[j];
         if (binding->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
             binding->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
            continue;
         if (binding->count > dynamic_count - dynamic) {
            rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
            return;
         }
         for (uint32_t k = 0; k < binding->count; k++)
            bound->dynamic_offsets[first + i][binding->slot + k] = offsets[dynamic++];
      }
   }
   if (dynamic != dynamic_count)
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdPushConstants(VkCommandBuffer handle, VkPipelineLayout layout,
   VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void *values)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (offset > 128 || size > 128 - offset) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Push constant values are shared by every stage and bind point. */
   (void)stages;
   memcpy((uint8_t *)command->push + offset, values, size);
}

static struct rd_dispatch *
rd_record_dispatch(struct rd_command_buffer *command)
{
   if (!command->pipeline) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }
   struct rd_command *recorded = rd_command_append(command, RD_CMD_DISPATCH);
   if (!recorded)
      return NULL;
   struct rd_dispatch *dispatch = &recorded->dispatch;
   dispatch->pipeline = command->pipeline;
   memcpy(dispatch->push, command->push, sizeof(dispatch->push));
   uint8_t compatibility[RD_MAX_SETS][BLAKE3_OUT_LEN];
   rd_layout_compatibility(command->pipeline->layout, compatibility);
   for (uint32_t i = 0; i < command->pipeline->resource_count; i++) {
      if (!(command->pipeline->used_descriptors & (UINT64_C(1) << i)))
         continue;
      const struct rd_nir_resource_binding *resource = &command->pipeline->resources[i];
      uint32_t index = resource->descriptor_set;
      struct rd_descriptor_set *set = command->bind[0].sets[index];
      if (!set || memcmp(command->bind[0].set_compatibility[index][index], compatibility[index], BLAKE3_OUT_LEN)) {
         rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
         return NULL;
      }
      dispatch->sets[index] = set;
      dispatch->dynamic_offsets[i] = command->bind[0].dynamic_offsets[index][resource->logical_slot];
   }
   return dispatch;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (!x || !y || !z)
      return;
   if (x > 65535 || y > 65535 || z > 65535) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct rd_dispatch *dispatch = rd_record_dispatch(command);
   if (!dispatch)
      return;
   dispatch->grid[0] = x;
   dispatch->grid[1] = y;
   dispatch->grid[2] = z;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDispatchBase(VkCommandBuffer handle, uint32_t bx, uint32_t by, uint32_t bz,
                   uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (bx >= 65535 || by >= 65535 || bz >= 65535 ||
       x > 65535 - bx || y > 65535 - by || z > 65535 - bz) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!x || !y || !z)
      return;
   struct rd_dispatch *dispatch = rd_record_dispatch(command);
   if (!dispatch)
      return;
   dispatch->base[0] = bx;
   dispatch->base[1] = by;
   dispatch->base[2] = bz;
   dispatch->grid[0] = x;
   dispatch->grid[1] = y;
   dispatch->grid[2] = z;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdExecuteCommands(VkCommandBuffer handle, uint32_t count,
                       const VkCommandBuffer *handles)
{
   VK_FROM_HANDLE(rd_command_buffer, primary, handle);
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(rd_command_buffer, secondary, handles[i]);
      util_dynarray_foreach(&secondary->commands, struct rd_command, source) {
         if (source->type == RD_CMD_RENDER) {
            /* Draws of a secondary recorded inside a render pass join the
             * primary's current render; a secondary's own renders are copied. */
            if (!rd_render_execute(primary, source->render))
               return;
            continue;
         }
         struct rd_command *destination = rd_command_append(primary, source->type);
         if (!destination)
            return;
         *destination = *source;
         if (source->type == RD_CMD_UPDATE) {
            destination->transfer.data = vk_alloc(&primary->vk.pool->alloc,
               source->transfer.size, 8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
            if (!destination->transfer.data) {
               rd_set_error(&primary->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
               return;
            }
            memcpy(destination->transfer.data, source->transfer.data,
                   source->transfer.size);
         }
      }
   }
   primary->pipeline = NULL;
   primary->gfx_pipeline = NULL;
   for (unsigned i = 0; i < 2; i++)
      memset(primary->bind[i].sets, 0, sizeof(primary->bind[i].sets));
   memset(primary->push, 0, sizeof(primary->push));
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDispatchIndirect(VkCommandBuffer handle, VkBuffer buffer_handle,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_buffer, buffer, buffer_handle);
   if (!buffer || !buffer->memory || (offset & 3) ||
       !(buffer->vk.usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) ||
       offset > buffer->vk.size ||
       sizeof(VkDispatchIndirectCommand) > buffer->vk.size - offset) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   struct rd_dispatch *dispatch = rd_record_dispatch(command);
   if (!dispatch)
      return;
   dispatch->indirect = buffer;
   dispatch->indirect_offset = offset;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdPipelineBarrier2(VkCommandBuffer handle, const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      if (!rd_image_barrier_valid(&info->pImageMemoryBarriers[i])) {
         if (debug_get_bool_option("RADISH_DEBUG_ERRORS", false)) {
            const VkImageMemoryBarrier2 *bar = &info->pImageMemoryBarriers[i];
            const struct rd_image *img = rd_image_from_handle(bar->image);
            fprintf(stderr, "radish: barrier refused: aspects %x/%x levels %u+%u/%u layers %u+%u/%u type %d\n",
                    bar->subresourceRange.aspectMask, img ? vk_format_aspects(img->vk.format) : 0,
                    bar->subresourceRange.baseMipLevel, bar->subresourceRange.levelCount,
                    img ? img->vk.mip_levels : 0, bar->subresourceRange.baseArrayLayer,
                    bar->subresourceRange.layerCount, img ? img->vk.array_layers : 0,
                    img ? img->vk.image_type : -1);
         }
         rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }
   /* inside a render pass (a subpass self-dependency): the next draw waits
    * for the attachment writes before it and reads them through textures */
   if (command->render)
      command->render_barrier = true;
}

struct rd_command *
rd_command_append(struct rd_command_buffer *command, enum rd_command_type type)
{
   struct rd_command *recorded = util_dynarray_grow_bytes(&command->commands,
      1, sizeof(struct rd_command));
   if (!recorded) {
      rd_set_error(&command->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return NULL;
   }
   memset(recorded, 0, sizeof(*recorded));
   recorded->type = type;
   return recorded;
}
