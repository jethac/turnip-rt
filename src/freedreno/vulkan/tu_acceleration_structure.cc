/*
 * Copyright © 2021 Bas Nieuwenhuizen
 * Copyright © 2024 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "tu_acceleration_structure.h"

#include "util/u_hexdump.h"
#include "vk_acceleration_structure.h"

#include "bvh/tu_bvh_defines.h"
#include "tu_buffer.h"
#include "tu_cmd_buffer.h"
#include "tu_device.h"

static const uint32_t encode_spv[] = {
#include "bvh/encode.spv.h"
};

static const uint32_t header_spv[] = {
#include "bvh/header.spv.h"
};

static const uint32_t copy_spv[] = {
#include "bvh/copy.spv.h"
};

static const uint32_t update_spv[] = {
#include "bvh/update.spv.h"
};

static const uint32_t update_batch_spv[] = {
#include "bvh/update_batch.spv.h"
};

static_assert(sizeof(struct tu_instance_descriptor) == AS_RECORD_SIZE);
static_assert(sizeof(struct tu_accel_struct_header) == AS_RECORD_SIZE);
static_assert(offsetof(struct tu_accel_struct_header, bvh_ptr) == 24);
static_assert(offsetof(struct tu_accel_struct_header, self_ptr) == 80);
static_assert(sizeof(struct tu_internal_node) == AS_NODE_SIZE);
static_assert(sizeof(struct tu_leaf_node) == AS_NODE_SIZE);
static_assert(sizeof(struct update_batch_task) == 96);
static_assert(offsetof(struct update_batch_task, geom_data) == 40);
static_assert(offsetof(struct update_batch_task, primitive_count) == 88);
static_assert(offsetof(struct update_batch_task, first_group) == 92);
static_assert(sizeof(struct update_batch_args) == 16);

static VkResult
get_pipeline_spv(struct tu_device *device,
                 const char *name, const uint32_t *spv, uint32_t spv_size,
                 unsigned push_constant_size,
                 VkPipeline *pipeline, VkPipelineLayout *layout)
{
   size_t key_size = strlen(name);

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_constant_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(&device->vk,
                                                 &device->meta, NULL,
                                                 &pc_range, name, key_size,
                                                 layout);

   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta, name, key_size);
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline = pipeline_from_cache;
      return VK_SUCCESS;
   }

   VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .codeSize = spv_size,
      .pCode = spv,
   };

   VkPipelineShaderStageCreateInfo shader_stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &module_info,
      .flags = 0,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .flags = 0,
      .stage = shader_stage,
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&device->vk, &device->meta, &pipeline_info,
                                          name, key_size, pipeline);
}

static uint32_t
tu_morton_sort_workgroup_size(void)
{
   static int value = -1;
   if (value < 0)
      value = debug_get_num_option("TU_MORTON_WG", 256);
   return value;
}

static uint32_t
tu_morton_sort_kvs_per_thread(void)
{
   static int value = -1;
   if (value < 0)
      value = debug_get_num_option("TU_MORTON_KVS", 2);
   return value;
}

static uint32_t
tu_as_subgroup_size(const struct tu_physical_device *pdev)
{
   static int value = -1;
   if (value < 0)
      value = debug_get_num_option("TU_AS_SUBGROUP", pdev->info->threadsize_base);
   uint32_t max = pdev->info->threadsize_base * (pdev->expose_double_threadsize ? 2 : 1);
   return CLAMP(util_next_power_of_two((uint32_t) value), (uint32_t) pdev->info->threadsize_base, max);
}

static bool
tu_as_use_hploc(void)
{
   static int value = -1;
   if (value < 0)
      value = debug_get_bool_option("TU_AS_HPLOC", true);
   return value;
}

static uint32_t
tu_rt_batch_refit(void)
{
   static int value = -1;
   if (value < 0)
      value = CLAMP(debug_get_num_option("TU_RT_BATCH_REFIT", 0), 0, 1);
   return value;
}

struct bvh_layout {
   uint64_t leaf_map_offset;
   uint64_t bvh_offset;
   uint64_t size;
};

static void
get_bvh_layout(VkGeometryTypeKHR geometry_type,
               uint32_t leaf_count,
               bool updateable,
               struct bvh_layout *layout)
{
   uint32_t internal_count = MAX2(leaf_count, 2) - 1;

   uint64_t offset = sizeof(struct tu_accel_struct_header);

   /* Instance descriptors, one per instance. */
   if (geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
      offset += leaf_count * sizeof(struct tu_instance_descriptor);
   }

   layout->leaf_map_offset = offset;
   if (updateable)
      offset += leaf_count * sizeof(uint32_t);

   /* Parent links, which have to go directly before bvh_offset as we index
    * them using negative offsets from there.
    */
   offset += (internal_count + leaf_count) * sizeof(uint32_t);

   /* The BVH and hence bvh_offset needs 64 byte alignment for RT nodes. */
   offset = align64(offset, 64);
   layout->bvh_offset = offset;

   offset += internal_count * sizeof(struct tu_internal_node) +
      leaf_count * sizeof(struct tu_leaf_node);

   layout->size = offset;
}

static VkDeviceSize
get_bvh_size(VkDevice device,
             const struct vk_acceleration_structure_build_state *state)
{
   struct bvh_layout layout;
   get_bvh_layout(vk_get_as_geometry_type(state->build_info),
                  state->leaf_node_count, state->config.updateable, &layout);
   return layout.size;
}

struct update_scratch_layout {
   uint64_t ready_count_offset;
   uint64_t bounds_offset;
   uint64_t size;
};

static void
get_update_scratch_layout(const struct vk_acceleration_structure_build_state *state,
                          struct update_scratch_layout *layout)
{
   uint32_t leaf_count = state->leaf_node_count;
   uint32_t node_count = MAX2(leaf_count, 2) - 1 + leaf_count;

   layout->ready_count_offset = 0;
   layout->bounds_offset = align64(node_count * sizeof(uint32_t), 64);
   layout->size = layout->bounds_offset + node_count * sizeof(vk_aabb);
}

static VkDeviceSize
tu_get_update_scratch_size(VkDevice device,
                           const struct vk_acceleration_structure_build_state *state)
{
   struct update_scratch_layout layout;
   get_update_scratch_layout(state, &layout);
   return layout.size;
}

static void
tu_init_update_scratch(VkCommandBuffer commandBuffer,
                       const struct vk_acceleration_structure_build_state *states,
                       uint32_t build_count)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmdbuf, commandBuffer);

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      struct update_scratch_layout layout;
      get_update_scratch_layout(state, &layout);

      cmdbuf->device->vk.cmd_fill_buffer_addr(commandBuffer,
                                              state->build_info->scratchData.deviceAddress +
                                                 layout.ready_count_offset,
                                              layout.bounds_offset - layout.ready_count_offset, 0);
   }
}

static void
tu_get_build_config(VkDevice device,
                    struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(tu_device, dev, device);
   if (tu_as_use_hploc() && tu_as_subgroup_size(dev->physical_device) <= 64 &&
       state->config.internal_type == VK_INTERNAL_BUILD_TYPE_PLOC) {
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_HPLOC;
      state->config.u64_keys = true;
   }

   static int log_builds = -1;
   if (log_builds < 0)
      log_builds = debug_get_bool_option("TU_AS_LOG", false);
   if (log_builds) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = state->build_info;
      mesa_logi("tu as build: type=%s mode=%s flags=0x%x leaves=%u internal=%u",
                info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ? "tlas" : "blas",
                info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR ? "update" : "build",
                info->flags, state->leaf_node_count, state->config.internal_type);
   }
}

static void
tu_update_as_batch_dispatch(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                            uint64_t tasks, uint32_t task_count, uint32_t group_count)
{
   const struct update_batch_args consts = {
      .tasks = tasks,
      .task_count = task_count,
      .group_count = group_count,
   };
   vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts), &consts);
   vk_common_CmdDispatch(commandBuffer, group_count, 1, 1);
}

static bool
tu_update_as_batched(VkCommandBuffer commandBuffer,
                     const struct vk_acceleration_structure_build_state *states,
                     uint32_t build_count)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmdbuf, commandBuffer);
   struct tu_device *device = cmdbuf->device;
   const uint32_t max_groups = 65535;

   uint32_t task_count = 0;
   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      for (uint32_t j = 0; j < state->build_info->geometryCount; j++) {
         uint32_t primitive_count = state->build_range_infos[j].primitiveCount;
         if (!primitive_count)
            continue;
         if (DIV_ROUND_UP(primitive_count, TU_UPDATE_WORKGROUP_SIZE) > max_groups)
            return false;
         task_count++;
      }
   }

   if (!task_count)
      return true;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result = get_pipeline_spv(device, "update_batch", update_batch_spv, sizeof(update_batch_spv),
                                      sizeof(update_batch_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return true;
   }

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   struct tu_cs_memory mem = {};
   uint32_t remaining = task_count;
   uint32_t chunk_tasks = 0;
   uint32_t chunk_groups = 0;

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      VkGeometryTypeKHR geometry_type = vk_get_as_geometry_type(state->build_info);

      struct bvh_layout bvh_layout;
      get_bvh_layout(geometry_type, state->leaf_node_count, true, &bvh_layout);

      struct update_scratch_layout scratch_layout;
      get_update_scratch_layout(state, &scratch_layout);

      VkDeviceAddress va = vk_acceleration_structure_get_va(dst);
      VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;

      uint32_t first_id = 0;
      for (uint32_t j = 0; j < state->build_info->geometryCount; j++) {
         const VkAccelerationStructureGeometryKHR *geom =
            state->build_info->pGeometries ? &state->build_info->pGeometries[j] : state->build_info->ppGeometries[j];
         const VkAccelerationStructureBuildRangeInfoKHR *range = &state->build_range_infos[j];
         uint32_t primitive_count = range->primitiveCount;

         if (primitive_count) {
            uint32_t groups = DIV_ROUND_UP(primitive_count, TU_UPDATE_WORKGROUP_SIZE);

            if (chunk_tasks && chunk_groups + groups > max_groups) {
               tu_update_as_batch_dispatch(commandBuffer, layout, mem.iova, chunk_tasks, chunk_groups);
               chunk_tasks = 0;
               chunk_groups = 0;
            }

            if (!chunk_tasks) {
               result = tu_cs_alloc(&cmdbuf->sub_cs,
                                    DIV_ROUND_UP(remaining * sizeof(struct update_batch_task), 64), 64 / 4,
                                    &mem);
               if (result != VK_SUCCESS) {
                  vk_command_buffer_set_error(&cmdbuf->vk, result);
                  return true;
               }
            }

            const struct update_batch_task task = {
               .bvh = va + bvh_layout.bvh_offset,
               .leaf_map = va + bvh_layout.leaf_map_offset,
               .header = va,
               .ready_count = scratch + scratch_layout.ready_count_offset,
               .bounds = scratch + scratch_layout.bounds_offset,
               .geom_data = vk_fill_geometry_data(state->build_info->type, first_id, j, geom, range),
               .primitive_count = primitive_count,
               .first_group = chunk_groups,
            };
            memcpy(reinterpret_cast<char *>(mem.map) + chunk_tasks * sizeof(task), &task, sizeof(task));

            chunk_tasks++;
            chunk_groups += groups;
            remaining--;
         }

         first_id += primitive_count;
      }
   }

   if (chunk_tasks)
      tu_update_as_batch_dispatch(commandBuffer, layout, mem.iova, chunk_tasks, chunk_groups);

   return true;
}

static void
tu_update_as(VkCommandBuffer commandBuffer,
             struct vk_acceleration_structure_build_state *states,
             uint32_t build_count)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmdbuf, commandBuffer);
   struct tu_device *device = cmdbuf->device;
   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result;

   bool need_copy = false;
   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE &&
          state->build_info->srcAccelerationStructure != state->build_info->dstAccelerationStructure)
         need_copy = true;
   }

   if (need_copy) {
      result = get_pipeline_spv(device, "copy", copy_spv, sizeof(copy_spv), sizeof(copy_args), &pipeline, &layout);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmdbuf->vk, result);
         return;
      }

      tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

      for (uint32_t i = 0; i < build_count; i++) {
         const struct vk_acceleration_structure_build_state *state = &states[i];
         if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
             state->build_info->srcAccelerationStructure == state->build_info->dstAccelerationStructure)
            continue;

         VK_FROM_HANDLE(vk_acceleration_structure, src, state->build_info->srcAccelerationStructure);
         VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);

         struct copy_args consts = {
            .src_addr = vk_acceleration_structure_get_va(src),
            .dst_addr = vk_acceleration_structure_get_va(dst),
            .mode = TU_COPY_MODE_COPY,
         };
         vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts), &consts);

         TU_CALLX(device, tu_CmdDispatchIndirect)(
            commandBuffer, vk_buffer_to_handle(src->buffer),
            src->offset + offsetof(struct tu_accel_struct_header, copy_dispatch_size));
      }
   }

   vk_bvh_build_barrier_transfer_to_compute(commandBuffer);
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   if (tu_rt_batch_refit() && tu_update_as_batched(commandBuffer, states, build_count))
      return;

   result = get_pipeline_spv(device, "update", update_spv, sizeof(update_spv), sizeof(update_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      VkGeometryTypeKHR geometry_type = vk_get_as_geometry_type(state->build_info);

      struct bvh_layout bvh_layout;
      get_bvh_layout(geometry_type, state->leaf_node_count, true, &bvh_layout);

      struct update_scratch_layout scratch_layout;
      get_update_scratch_layout(state, &scratch_layout);

      VkDeviceAddress va = vk_acceleration_structure_get_va(dst);
      VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;

      struct update_args consts = {
         .bvh = va + bvh_layout.bvh_offset,
         .leaf_map = va + bvh_layout.leaf_map_offset,
         .header = va,
         .ready_count = scratch + scratch_layout.ready_count_offset,
         .bounds = scratch + scratch_layout.bounds_offset,
      };

      uint32_t first_id = 0;
      for (uint32_t j = 0; j < state->build_info->geometryCount; j++) {
         const VkAccelerationStructureGeometryKHR *geom =
            state->build_info->pGeometries ? &state->build_info->pGeometries[j] : state->build_info->ppGeometries[j];
         const VkAccelerationStructureBuildRangeInfoKHR *range = &state->build_range_infos[j];

         consts.geom_data = vk_fill_geometry_data(state->build_info->type, first_id, j, geom, range);
         consts.primitive_count = range->primitiveCount;

         vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts), &consts);
         tu_dispatch_unaligned(commandBuffer, range->primitiveCount, 1, 1);

         first_id += range->primitiveCount;
      }
   }
}

static void
tu_encode(VkCommandBuffer commandBuffer,
          struct vk_device *vk_device,
          struct vk_meta_device *meta,
          const struct vk_acceleration_structure_build_args *args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmdbuf, commandBuffer);
   struct tu_device *device = cmdbuf->device;
   VkResult result;
   VkPipeline pipeline;
   VkPipelineLayout layout;

   bool has_update = false, has_build = false;
   for (uint32_t i = 0; i < build_count; i++) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         has_update = true;
      else
         has_build = true;
   }

   if (has_update)
      tu_update_as(commandBuffer, states, build_count);

   if (!has_build)
      return;

   /* Insert barrier so that build data gets reflected properly for encode pass */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, true);

   result = get_pipeline_spv(device, "encode", encode_spv, sizeof(encode_spv), sizeof(encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   bool has_compaction = false;
   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      if (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
         has_compaction = true;

      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      VkGeometryTypeKHR geometry_type = vk_get_as_geometry_type(state->build_info);

      uint64_t intermediate_header_addr = state->build_info->scratchData.deviceAddress + state->scratch.header_offset;
      uint64_t intermediate_bvh_addr = state->build_info->scratchData.deviceAddress + state->scratch.ir_offset;

      struct bvh_layout bvh_layout;
      get_bvh_layout(geometry_type, state->leaf_node_count, state->config.updateable, &bvh_layout);

      const struct encode_args args = {
         .intermediate_bvh = intermediate_bvh_addr,
         .output_bvh = vk_acceleration_structure_get_va(dst) + bvh_layout.bvh_offset,
         .header = intermediate_header_addr,
         .leaf_map = state->config.updateable
                        ? vk_acceleration_structure_get_va(dst) + bvh_layout.leaf_map_offset
                        : 0,
         .output_bvh_offset = bvh_layout.bvh_offset,
         .leaf_node_count = state->leaf_node_count,
         .geometry_type = geometry_type,
      };
      vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args), &args);

      tu_dispatch_unaligned_indirect(commandBuffer,
                                     intermediate_header_addr + offsetof(struct vk_ir_header, ir_internal_node_count));
   }

   if (has_compaction) {
      result =
         get_pipeline_spv(device, "header", header_spv, sizeof(header_spv), sizeof(header_args), &pipeline, &layout);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmdbuf->vk, result);
         return;
      }

      static const VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      };

      vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

      tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   }

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      VkGeometryTypeKHR geometry_type = vk_get_as_geometry_type(state->build_info);

      struct bvh_layout bvh_layout;
      get_bvh_layout(geometry_type, state->leaf_node_count, state->config.updateable, &bvh_layout);

      uint64_t intermediate_header_addr = state->build_info->scratchData.deviceAddress + state->scratch.header_offset;
      VkDeviceAddress header_addr = vk_acceleration_structure_get_va(dst);

      size_t base = offsetof(struct tu_accel_struct_header, copy_dispatch_size);

      uint32_t instance_count = geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR ? state->leaf_node_count : 0;

      if (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) {
         base = offsetof(struct tu_accel_struct_header, instance_count);

         struct header_args args = {
            .src = intermediate_header_addr,
            .dst = vk_acceleration_structure_get_va(dst),
            .bvh_offset = bvh_layout.bvh_offset,
            .instance_count = instance_count,
         };

         vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args), &args);

         vk_common_CmdDispatch(commandBuffer, 1, 1, 1);
      }

      struct tu_accel_struct_header header = {};

      header.instance_count = instance_count;
      header.self_ptr = header_addr;
      header.compacted_size = bvh_layout.size;

      header.copy_dispatch_size[0] = DIV_ROUND_UP(header.compacted_size, 16 * 128);
      header.copy_dispatch_size[1] = 1;
      header.copy_dispatch_size[2] = 1;

      header.serialization_size = header.compacted_size + sizeof(struct vk_accel_struct_serialization_header) +
                                  sizeof(uint64_t) * header.instance_count;

      header.size = header.serialization_size - sizeof(struct vk_accel_struct_serialization_header) -
                    sizeof(uint64_t) * header.instance_count;

      struct tu_cs *cs = &cmdbuf->cs;

      size_t header_size = sizeof(struct tu_accel_struct_header) - base;
      assert(base % sizeof(uint32_t) == 0);
      assert(header_size % sizeof(uint32_t) == 0);
      uint32_t *header_ptr = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(&header) + base);

      tu_cs_emit_pkt7(cs, CP_MEM_WRITE, 2 + header_size / sizeof(uint32_t));
      tu_cs_emit_qw(cs, header_addr + base);
      tu_cs_emit_array(cs, header_ptr, header_size / sizeof(uint32_t));
   }
}

const struct vk_acceleration_structure_build_ops tu_as_build_ops = {
   .get_build_config = tu_get_build_config,
   .get_as_size = get_bvh_size,
   .get_update_scratch_size = tu_get_update_scratch_size,
   .init_update_scratch = tu_init_update_scratch,
   .encode = tu_encode,
};

struct tu_saved_compute_state {
   uint32_t push_constants[MAX_PUSH_CONSTANTS_SIZE / 4];
   struct tu_shader *compute_shader;
};

static void
tu_save_compute_state(struct tu_cmd_buffer *cmd,
                      struct tu_saved_compute_state *state)
{
   memcpy(state->push_constants, cmd->push_constants, sizeof(cmd->push_constants));
   state->compute_shader = cmd->state.shaders[MESA_SHADER_COMPUTE];
}

static void
tu_restore_compute_state(struct tu_cmd_buffer *cmd,
                         struct tu_saved_compute_state *state)
{
   cmd->state.shaders[MESA_SHADER_COMPUTE] = state->compute_shader;
   if (state->compute_shader) {
      tu_cs_emit_state_ib(&cmd->cs, state->compute_shader->state);
   }
   memcpy(cmd->push_constants, state->push_constants, sizeof(cmd->push_constants));
   cmd->state.dirty |= TU_CMD_DIRTY_SHADER_CONSTS;
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdBuildAccelerationStructuresKHR(VkCommandBuffer commandBuffer, uint32_t infoCount,
                                     const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
                                     const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   struct tu_device *device = cmd->device;
   struct tu_saved_compute_state state;

   tu_save_compute_state(cmd, &state);

   struct vk_acceleration_structure_build_args args = {
      .subgroup_size = tu_as_subgroup_size(device->physical_device),
      .bvh_bounds_offset = offsetof(tu_accel_struct_header, aabb),
      .morton_sort_workgroup_size = tu_morton_sort_workgroup_size(),
      .morton_sort_kvs_per_thread = tu_morton_sort_kvs_per_thread(),
      .emit_markers = false,
      .has_update = true,
   };

   vk_cmd_build_acceleration_structures(commandBuffer,
                                        &device->vk,
                                        &device->meta,
                                        infoCount,
                                        pInfos,
                                        ppBuildRangeInfos,
                                        &args);

   tu_restore_compute_state(cmd, &state);
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdCopyAccelerationStructureKHR(VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);
   struct tu_saved_compute_state state;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(cmd->device, "copy", copy_spv, sizeof(copy_spv),
                    sizeof(copy_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   tu_save_compute_state(cmd, &state);

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   struct copy_args consts = {
      .src_addr = vk_acceleration_structure_get_va(src),
      .dst_addr = vk_acceleration_structure_get_va(dst),
      .mode = TU_COPY_MODE_COPY,
   };

   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts),
                              &consts);

   TU_CALLX(cmd->device, tu_CmdDispatchIndirect)(
      commandBuffer, vk_buffer_to_handle(src->buffer),
      src->offset + offsetof(struct tu_accel_struct_header, copy_dispatch_size));

   tu_restore_compute_state(cmd, &state);
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdCopyMemoryToAccelerationStructureKHR(VkCommandBuffer commandBuffer,
                                           const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);
   struct tu_saved_compute_state state;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(cmd->device, "copy", copy_spv, sizeof(copy_spv),
                    sizeof(copy_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   tu_save_compute_state(cmd, &state);

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   const struct copy_args consts = {
      .src_addr = pInfo->src.deviceAddress,
      .dst_addr = vk_acceleration_structure_get_va(dst),
      .mode = TU_COPY_MODE_DESERIALIZE,
   };

   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts),
                              &consts);

   vk_common_CmdDispatch(commandBuffer, 256, 1, 1);

   tu_restore_compute_state(cmd, &state);
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdCopyAccelerationStructureToMemoryKHR(VkCommandBuffer commandBuffer,
                                           const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(tu_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);
   struct tu_saved_compute_state state;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(cmd->device, "copy", copy_spv, sizeof(copy_spv),
                    sizeof(copy_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   tu_save_compute_state(cmd, &state);

   tu_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   const struct copy_args consts = {
      .src_addr = vk_acceleration_structure_get_va(src),
      .dst_addr = pInfo->dst.deviceAddress,
      .mode = TU_COPY_MODE_SERIALIZE,
   };

   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(consts),
                              &consts);

   TU_CALLX(cmd->device, tu_CmdDispatchIndirect)(
      commandBuffer, vk_buffer_to_handle(src->buffer),
      src->offset + offsetof(struct tu_accel_struct_header, copy_dispatch_size));

   tu_restore_compute_state(cmd, &state);

   /* Set the header of the serialized data. */
   uint32_t header_data[2 * VK_UUID_SIZE / 4];
   memcpy(header_data, cmd->device->physical_device->driver_uuid, VK_UUID_SIZE);
   memcpy(header_data + VK_UUID_SIZE / 4, cmd->device->physical_device->cache_uuid, VK_UUID_SIZE);

   struct tu_cs *cs = &cmd->cs;

   tu_cs_emit_pkt7(cs, CP_MEM_WRITE, 2 + ARRAY_SIZE(header_data));
   tu_cs_emit_qw(cs, pInfo->dst.deviceAddress);
   tu_cs_emit_array(cs, header_data, ARRAY_SIZE(header_data));
}

VKAPI_ATTR void VKAPI_CALL
tu_GetAccelerationStructureBuildSizesKHR(VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
                                         const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
                                         const uint32_t *pMaxPrimitiveCounts,
                                         VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(tu_device, device, _device);

   struct vk_acceleration_structure_build_args args = {
      .subgroup_size = tu_as_subgroup_size(device->physical_device),
      .morton_sort_workgroup_size = tu_morton_sort_workgroup_size(),
      .morton_sort_kvs_per_thread = tu_morton_sort_kvs_per_thread(),
      .has_update = true,
   };

   vk_get_as_build_sizes(_device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &args);
   if (pSizeInfo->accelerationStructureSize == 0 || pSizeInfo->buildScratchSize == 0) {
      uint32_t prims = 0;
      for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++)
         prims += pMaxPrimitiveCounts[i];
      mesa_logw("tu: AS build sizes type=%u geoms=%u prims=%u -> as=%" PRIu64
                " scratch=%" PRIu64 " update=%" PRIu64,
                pBuildInfo->type, pBuildInfo->geometryCount, prims,
                (uint64_t) pSizeInfo->accelerationStructureSize,
                (uint64_t) pSizeInfo->buildScratchSize,
                (uint64_t) pSizeInfo->updateScratchSize);
   }
}

VKAPI_ATTR void VKAPI_CALL
tu_GetDeviceAccelerationStructureCompatibilityKHR(VkDevice _device,
                                                  const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
                                                  VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(tu_device, device, _device);
   bool compat =
      memcmp(pVersionInfo->pVersionData, device->physical_device->driver_uuid, VK_UUID_SIZE) == 0 &&
      memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE, device->physical_device->cache_uuid, VK_UUID_SIZE) == 0;
   *pCompatibility = compat ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                            : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VkResult
tu_init_null_accel_struct(struct tu_device *device)
{
   VkResult result = tu_bo_init_new(device, NULL,
                                    &device->null_accel_struct_bo,
                                    sizeof(tu_accel_struct_header) +
                                    sizeof(tu_internal_node),
                                    TU_BO_ALLOC_NO_FLAGS, "null AS");
   if (result != VK_SUCCESS) {
      return result;
   }

   result = tu_bo_map(device, device->null_accel_struct_bo, NULL);
   if (result != VK_SUCCESS) {
      tu_bo_finish(device, device->null_accel_struct_bo);
      return result;
   }

   struct tu_accel_struct_header header = {
      .bvh_ptr = device->null_accel_struct_bo->iova +
         sizeof(tu_accel_struct_header),
      .self_ptr = device->null_accel_struct_bo->iova,
   };

   struct tu_internal_node node = {
      .child_count = 0,
      .type_flags = 0,
   };

   for (unsigned i = 0; i < 8; i++) {
      node.mantissas[i][0][0] = 0xff;
      node.mantissas[i][0][1] = 0xff;
      node.mantissas[i][0][2] = 0xff;
   }

   memcpy(device->null_accel_struct_bo->map, &header, sizeof(header));
   memcpy(static_cast<char *>(device->null_accel_struct_bo->map) + sizeof(header),
          &node, sizeof(node));
   return VK_SUCCESS;
}

struct tu_node {
   uint32_t data[16];
};

static void
dump_leaf(struct tu_leaf_node *node)
{
   fprintf(stderr, "\tID: %d\n", node->id);
   fprintf(stderr, "\tgeometry ID: %d\n", node->geometry_id);
   bool aabb = node->type_flags & TU_NODE_TYPE_AABB;
   for (unsigned i = 0; i < (aabb ? 2 : 3); i++) {
      fprintf(stderr, "\t(");
      for (unsigned j = 0; j < 3; j++) {
         if (j != 0)
            fprintf(stderr, ", ");
         fprintf(stderr, "%f", node->coords[i][j]);
      }
      fprintf(stderr, ")\n");
   }
}

static void
dump_internal(struct tu_internal_node *node, uint32_t *max_child)
{
   *max_child = MAX2(*max_child, node->id + node->child_count);
   float base[3];
   unsigned exponents[3];
   for (unsigned i = 0; i < 3; i++) {
      base[i] = uif(node->bases[i] << 16);
      exponents[i] = node->exponents[i] - 134;
   }

   for (unsigned i = 0; i < node->child_count; i++) {
      fprintf(stderr, "\tchild %d\n", node->id + i);
      for (unsigned vert = 0; vert < 2; vert++) {
         fprintf(stderr, "\t\t(");
         for (unsigned coord = 0; coord < 3; coord++) {
            unsigned mantissa = node->mantissas[i][vert][coord];
            if (coord != 0)
               fprintf(stderr, ", ");
            fprintf(stderr, "%f", base[coord] + ldexp(static_cast<float>(mantissa),
                                                      exponents[coord]));
         }
         fprintf(stderr, ")\n");
      }
   }
}

static void
dump_as(struct vk_acceleration_structure *as)
{
   VK_FROM_HANDLE(tu_buffer, buf, vk_buffer_to_handle(as->buffer));

   struct tu_accel_struct_header *hdr =
      reinterpret_cast<struct tu_accel_struct_header *>(static_cast<char *>(buf->bo->map) + as->offset);

   fprintf(stderr, "dumping AS at %" PRIx64 "\n",
           vk_buffer_address(&buf->vk, as->offset));
   u_hexdump(stderr, reinterpret_cast<uint8_t *>(hdr), sizeof(*hdr), false);

   char *base = (static_cast<char *>(buf->bo->map) + (hdr->bvh_ptr - buf->vk.device_address));
   struct tu_node *node = reinterpret_cast<struct tu_node *>(base);

   fprintf(stderr, "dumping nodes at %" PRIx64 "\n", hdr->bvh_ptr);

   uint32_t max_child = 1;
   for (unsigned i = 0; i < max_child; i++) {
      uint32_t *parent_ptr = reinterpret_cast<uint32_t*>(base - (4 + 4 * i));
      uint32_t parent = *parent_ptr;
      fprintf(stderr, "node %d parent %d\n", i, parent);
      u_hexdump(stderr, reinterpret_cast<uint8_t *>(node), sizeof(*node), false);
      if (node->data[15] & TU_NODE_TYPE_LEAF) {
         /* TODO compressed leaves */
         dump_leaf(reinterpret_cast<struct tu_leaf_node *>(node));
      } else {
         dump_internal(reinterpret_cast<struct tu_internal_node *>(node), &max_child);
      }

      node++;
   }
}

static bool
as_finished(struct tu_device *dev, struct vk_acceleration_structure *as)
{
   VK_FROM_HANDLE(tu_buffer, buf, vk_buffer_to_handle(as->buffer));
   tu_bo_map(dev, buf->bo, NULL);

   struct tu_accel_struct_header *hdr =
      reinterpret_cast<struct tu_accel_struct_header *>(static_cast<char *>(buf->bo->map) + as->offset);
   return hdr->self_ptr == vk_buffer_address(&buf->vk, as->offset);
}

VKAPI_ATTR void VKAPI_CALL
tu_DestroyAccelerationStructureKHR(VkDevice _device,
                                   VkAccelerationStructureKHR accelerationStructure,
                                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(tu_device, device, _device);
   if (TU_DEBUG(DUMPAS)) {
      VK_FROM_HANDLE(vk_acceleration_structure, as, accelerationStructure);
      if (as_finished(device, as))
         dump_as(as);
   }

   vk_common_DestroyAccelerationStructureKHR(_device, accelerationStructure,
                                             pAllocator);
}
