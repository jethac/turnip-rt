/*
 * Radish graphics: dynamic rendering in bypass (sysmem) mode, graphics
 * pipelines with ir3 vertex and fragment shaders, and draws.
 *
 * Graphics exists for Vulkan 1.1 conformance; the GPU budget at run time is
 * compute. The register programming follows Freedreno's a5xx Gallium driver
 * (src/gallium/drivers/freedreno/a5xx: fd5_emit.c, fd5_gmem.c,
 * fd5_program.c, fd5_draw.c), Copyright © 2016 Rob Clark, MIT.
 *
 * Work flow: vkCmdBeginRendering opens an rd_render recorded as one
 * RD_CMD_RENDER; draws append to it; at submit the whole render instance is
 * emitted into one ring and run synchronously, like a compute dispatch.
 */
#include "rd_private.h"
#include "rd_format.h"
#include "common/fd5_hw.h"
#include "drm/freedreno_ringbuffer.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"

#define A5XX_MAX_RENDER_TARGETS 8
#include "ir3/ir3_nir.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_format.h"
#include "vk_util.h"
#include "util/u_debug.h"

bool
rd_graphics_enabled(void)
{
   return debug_get_bool_option("RADISH_GRAPHICS", false);
}

/* --- pipelines ------------------------------------------------------------ */

static VkResult
rd_gfx_upload(struct rd_device *device, struct rd_gfx_shader *shader)
{
   struct ir3_shader_variant *v = shader->v;
   if (v->pvtmem_size || v->info.size > UINT32_MAX - 127)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   shader->bo = fd_bo_new(device->fd, align(v->info.size, 128), FD_BO_GPUREADONLY,
                          "radish-gfx-shader");
   if (!shader->bo)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   void *map = fd_bo_map(shader->bo);
   if (!map)
      return VK_ERROR_MEMORY_MAP_FAILED;
   memcpy(map, v->bin, v->info.size);
   return VK_SUCCESS;
}

/* Buffers in graphics stages are read through global loads from addresses
 * the driver puts in the const file after the push constants (graphics
 * stages have no SSBO state on a5xx's vertex side). Stores and atomics need
 * vertexPipelineStoresAndAtomics / fragmentStoresAndAtomics, which are not
 * advertised.
 */
#define RD_GFX_PUSH_DWORDS 32
#define RD_GFX_MAX_BUFFERS 16
#define RD_GFX_CONST_VEC4 ((RD_GFX_PUSH_DWORDS + 2 * RD_GFX_MAX_BUFFERS) / 4)

struct rd_gfx_lower { bool failed; };

static bool
rd_gfx_lower_buffer(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rd_gfx_lower *state = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ssbo: {
      if (!nir_src_is_const(intr->src[0]) || intr->def.bit_size != 32 ||
          nir_src_as_uint(intr->src[0]) >= RD_GFX_MAX_BUFFERS) {
         state->failed = true;
         return false;
      }
      b->cursor = nir_before_instr(&intr->instr);
      unsigned slot = nir_src_as_uint(intr->src[0]);
      nir_def *address = nir_load_const_ir3(b, 2, 32, nir_imm_int(b, 0),
                                            .base = RD_GFX_PUSH_DWORDS + 2 * slot);
      nir_def *value = nir_load_global_offset(b, intr->def.num_components, 32,
                                              nir_pack_64_2x32(b, address), intr->src[1].ssa,
                                              .align_mul = nir_intrinsic_align_mul(intr),
                                              .align_offset = nir_intrinsic_align_offset(intr));
      nir_def_replace(&intr->def, value);
      return true;
   }
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_get_ssbo_size:
      state->failed = true;
      return false;
   default:
      if (!strncmp(nir_intrinsic_infos[intr->intrinsic].name, "image_", 6))
         state->failed = true;
      return false;
   }
}

static VkResult
rd_gfx_compile(struct rd_device *device, struct rd_pipeline *pipeline,
               struct rd_gfx_shader *shader,
               const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags)
{
   const struct spirv_to_nir_options spirv_options = {
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_logical,
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ubo_alignment = 16, .min_ssbo_alignment = 4,
   };
   nir_shader *nir = NULL;
   VkResult result = vk_pipeline_shader_stage_to_nir(&device->vk, flags, stage,
      &spirv_options, ir3_get_compiler_options(device->compiler), NULL, &nir);
   if (result != VK_SUCCESS)
      return result;
   shader->res = vk_zalloc(&device->vk.alloc, sizeof(struct rd_pipeline), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!shader->res) {
      ralloc_free(nir);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   shader->res->layout = pipeline->layout;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);
   struct rd_gfx_lower lower = {0};
   if (!rd_nir_lower_compute_resources(nir, rd_pipeline_resource, shader->res,
                                       &shader->res->resource_usage) ||
       !rd_lower_stage_buffers(nir, shader->res) ||
       shader->res->resource_usage.image_count ||
       shader->res->buffer_count > RD_GFX_MAX_BUFFERS) {
      ralloc_free(nir);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   NIR_PASS(_, nir, nir_opt_constant_folding);
   nir_shader_intrinsics_pass(nir, rd_gfx_lower_buffer, nir_metadata_control_flow, &lower);
   if (lower.failed) {
      ralloc_free(nir);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   NIR_PASS(_, nir, nir_opt_dce);
   ir3_nir_lower_io_vars_to_temporaries(nir);
   nir_assign_io_var_locations(nir, nir_var_shader_in);
   nir_assign_io_var_locations(nir, nir_var_shader_out);
   ir3_nir_lower_io(nir);
   struct ir3_shader_options options = {
      .nir_options.disable_optimizations = flags & VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT,
      .nir_options.a5xx_raw_global_loads = true,
      .api_wavesize = IR3_SINGLE_ONLY,
      .real_wavesize = IR3_SINGLE_ONLY,
      .push_consts_type = IR3_PUSH_CONSTS_PER_STAGE,
      .push_consts_dwords = RD_GFX_PUSH_DWORDS + 2 * RD_GFX_MAX_BUFFERS,
      .const_allocs = {
         .consts[IR3_CONST_ALLOC_PUSH_CONSTS] = { .offset_vec4 = 0, .size_vec4 = RD_GFX_CONST_VEC4 },
         .max_const_offset_vec4 = RD_GFX_CONST_VEC4,
      },
   };
   ir3_finalize_nir(device->compiler, &options.nir_options, nir);
   struct ir3_shader *ir3 = ir3_shader_from_nir(device->compiler, nir, &options);
   if (!ir3)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct ir3_shader_key key = {0};
   shader->v = ir3_shader_create_variant(ir3, &key, false);
   ir3_shader_destroy(ir3);
   if (!shader->v)
      return VK_ERROR_UNKNOWN;
   shader->v->shader = NULL;
   return rd_gfx_upload(device, shader);
}

void
rd_gfx_pipeline_destroy(struct rd_device *device, struct rd_pipeline *pipeline)
{
   struct rd_gfx_pipeline *gfx = pipeline->gfx;
   if (!gfx)
      return;
   struct rd_gfx_shader *shaders[] = { &gfx->vs, &gfx->fs };
   for (unsigned i = 0; i < 2; i++) {
      if (shaders[i]->bo)
         fd_bo_del(shaders[i]->bo);
      if (shaders[i]->v)
         ralloc_free(shaders[i]->v);
      vk_free(&device->vk.alloc, shaders[i]->res);
   }
   vk_free(&device->vk.alloc, gfx);
   pipeline->gfx = NULL;
}

void rd_destroy_pipeline_public(struct rd_device *device, struct rd_pipeline *pipeline,
                                const VkAllocationCallbacks *allocator);

static VkResult
rd_create_graphics_pipeline(struct rd_device *device, const VkGraphicsPipelineCreateInfo *info,
                            const VkAllocationCallbacks *allocator, VkPipeline *out)
{
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   struct rd_pipeline *pipeline = vk_object_zalloc(&device->vk, allocator, sizeof(*pipeline),
                                                   VK_OBJECT_TYPE_PIPELINE);
   if (!pipeline)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   pipeline->layout = vk_pipeline_layout_ref(layout);
   struct rd_gfx_pipeline *gfx = vk_zalloc(&device->vk.alloc, sizeof(*gfx), 8,
                                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->gfx = gfx;
   VkResult result = gfx ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result == VK_SUCCESS)
      result = vk_graphics_pipeline_state_fill(&device->vk, &gfx->state, info, NULL, NULL, 0,
                                               &gfx->all, NULL, 0, NULL);
   const VkPipelineShaderStageCreateInfo *vs = NULL, *fs = NULL;
   for (uint32_t i = 0; result == VK_SUCCESS && i < info->stageCount; i++) {
      if (info->pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT)
         vs = &info->pStages[i];
      else if (info->pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
         fs = &info->pStages[i];
      else
         result = VK_ERROR_FEATURE_NOT_PRESENT;
   }
   /* Fragment-less pipelines (depth-only) come later. */
   if (result == VK_SUCCESS && (!vs || !fs))
      result = VK_ERROR_FEATURE_NOT_PRESENT;
   if (result == VK_SUCCESS)
      result = rd_gfx_compile(device, pipeline, &gfx->vs, vs, info->flags);
   if (result == VK_SUCCESS)
      result = rd_gfx_compile(device, pipeline, &gfx->fs, fs, info->flags);
   if (result == VK_SUCCESS) {
      vk_dynamic_graphics_state_init(&gfx->dynamic);
      vk_dynamic_graphics_state_fill(&gfx->dynamic, &gfx->state);
   }
   if (result != VK_SUCCESS) {
      rd_destroy_pipeline_public(device, pipeline, allocator);
      return result;
   }
   *out = rd_pipeline_to_handle(pipeline);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateGraphicsPipelines(VkDevice handle, VkPipelineCache cache, uint32_t count,
   const VkGraphicsPipelineCreateInfo *infos, const VkAllocationCallbacks *allocator,
   VkPipeline *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VkResult failure = VK_SUCCESS;
   memset(out, 0, count * sizeof(*out));
   for (uint32_t i = 0; i < count; i++) {
      VkResult result = rd_graphics_enabled() ?
         rd_create_graphics_pipeline(device, &infos[i], allocator, &out[i]) :
         VK_ERROR_FEATURE_NOT_PRESENT;
      if (result != VK_SUCCESS)
         failure = result;
   }
   return failure;
}

/* --- recording ------------------------------------------------------------ */

void
rd_render_free(struct rd_command_buffer *command, struct rd_render *render)
{
   if (!render)
      return;
   util_dynarray_foreach(&render->draws, struct rd_draw, draw) {
      vk_free(&command->vk.pool->alloc, draw->descriptors[0]);
      vk_free(&command->vk.pool->alloc, draw->descriptors[1]);
   }
   util_dynarray_fini(&render->draws);
   vk_free(&command->vk.pool->alloc, render);
}

static void
rd_render_attachment(struct rd_render_attachment *out, const VkRenderingAttachmentInfo *info)
{
   if (!info || !info->imageView)
      return;
   out->view = rd_image_view_from_handle(info->imageView);
   out->load = info->loadOp;
   out->store = info->storeOp;
   out->clear = info->clearValue;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBeginRendering(VkCommandBuffer handle, const VkRenderingInfo *info)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (command->render || info->colorAttachmentCount > RD_MAX_RTS ||
       (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT))) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct rd_render *render = vk_zalloc(&command->vk.pool->alloc, sizeof(*render), 8,
                                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!render) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   util_dynarray_init(&render->draws, NULL);
   render->area = info->renderArea;
   render->layers = info->layerCount;
   render->view_mask = info->viewMask;
   render->color_count = info->colorAttachmentCount;
   for (uint32_t i = 0; i < info->colorAttachmentCount; i++)
      rd_render_attachment(&render->color[i], &info->pColorAttachments[i]);
   rd_render_attachment(&render->depth, info->pDepthAttachment);
   rd_render_attachment(&render->stencil, info->pStencilAttachment);
   struct rd_command *recorded = rd_command_append(command, RD_CMD_RENDER);
   if (!recorded) {
      rd_render_free(command, render);
      return;
   }
   recorded->render = render;
   command->render = render;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdEndRendering(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   command->render = NULL;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBindVertexBuffers2(VkCommandBuffer handle, uint32_t first, uint32_t count,
                         const VkBuffer *buffers, const VkDeviceSize *offsets,
                         const VkDeviceSize *sizes, const VkDeviceSize *strides)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (first + count > 32 || strides) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   for (uint32_t i = 0; i < count; i++) {
      command->vertex_buffers[first + i] = rd_buffer_from_handle(buffers[i]);
      command->vertex_offsets[first + i] = offsets[i];
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBindVertexBuffers(VkCommandBuffer handle, uint32_t first, uint32_t count,
                        const VkBuffer *buffers, const VkDeviceSize *offsets)
{
   rd_CmdBindVertexBuffers2(handle, first, count, buffers, offsets, NULL, NULL);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBindIndexBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                      VkIndexType type)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   command->index_buffer = rd_buffer_from_handle(buffer);
   command->index_offset = offset;
   command->index_type = type;
}

static bool
rd_copy_stage_descriptors(struct rd_command_buffer *command, const struct rd_pipeline *stage,
                          struct rd_descriptor **out)
{
   *out = NULL;
   if (!stage->resource_count)
      return true;
   struct rd_descriptor *copy = vk_zalloc(&command->vk.pool->alloc,
      stage->resource_count * sizeof(*copy), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!copy)
      return false;
   const struct rd_bind_state *bound = &command->bind[1];
   for (uint32_t i = 0; i < stage->resource_count; i++) {
      const struct rd_nir_resource_binding *resource = &stage->resources[i];
      const struct rd_descriptor_set *set = bound->sets[resource->descriptor_set];
      if (!set) {
         vk_free(&command->vk.pool->alloc, copy);
         return false;
      }
      copy[i] = set->descriptors[resource->logical_slot];
      if (copy[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          copy[i].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         copy[i].offset += bound->dynamic_offsets[resource->descriptor_set][resource->logical_slot];
   }
   *out = copy;
   return true;
}

static struct rd_render *
rd_render_new(struct rd_command_buffer *command)
{
   struct rd_render *render = vk_zalloc(&command->vk.pool->alloc, sizeof(*render), 8,
                                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!render) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return NULL;
   }
   util_dynarray_init(&render->draws, NULL);
   struct rd_command *recorded = rd_command_append(command, RD_CMD_RENDER);
   if (!recorded) {
      rd_render_free(command, render);
      return NULL;
   }
   recorded->render = render;
   return render;
}

static struct rd_descriptor *
rd_clone_descriptors(struct rd_command_buffer *command, const struct rd_descriptor *src,
                     uint32_t count)
{
   if (!src || !count)
      return NULL;
   struct rd_descriptor *copy = vk_alloc(&command->vk.pool->alloc, count * sizeof(*copy), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (copy)
      memcpy(copy, src, count * sizeof(*copy));
   return copy;
}

bool
rd_render_execute(struct rd_command_buffer *primary, const struct rd_render *src)
{
   struct rd_render *dst = primary->render;
   if (!src->continued) {
      dst = rd_render_new(primary);
      if (!dst)
         return false;
      struct util_dynarray draws = dst->draws;
      *dst = *src;
      dst->draws = draws;
   } else if (!dst) {
      vk_command_buffer_set_error(&primary->vk, VK_ERROR_UNKNOWN);
      return false;
   }
   util_dynarray_foreach(&src->draws, struct rd_draw, draw) {
      struct rd_draw copy = *draw;
      copy.descriptors[0] = rd_clone_descriptors(primary, draw->descriptors[0],
                                                 draw->pipeline->gfx->vs.res->resource_count);
      copy.descriptors[1] = rd_clone_descriptors(primary, draw->descriptors[1],
                                                 draw->pipeline->gfx->fs.res->resource_count);
      util_dynarray_append(&dst->draws, copy);
   }
   return true;
}

static struct rd_draw *
rd_record_draw(struct rd_command_buffer *command)
{
   struct rd_pipeline *pipeline = command->gfx_pipeline;
   if (!command->render && command->render_continue) {
      command->render = rd_render_new(command);
      if (!command->render)
         return NULL;
      command->render->continued = true;
   }
   if (!command->render || !pipeline || !pipeline->gfx) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }
   const struct vk_dynamic_graphics_state *dyn = &command->vk.dynamic_graphics_state;
   struct rd_draw draw = {
      .pipeline = pipeline,
      .viewport = dyn->vp.viewports[0],
      .scissor = dyn->vp.scissors[0],
      .cull = dyn->rs.cull_mode,
      .front = dyn->rs.front_face,
      .ds = dyn->ds,
      .cb = dyn->cb,
      .bias_enable = dyn->rs.depth_bias.enable,
      .bias_constant = dyn->rs.depth_bias.constant_factor,
      .bias_clamp = dyn->rs.depth_bias.clamp,
      .bias_slope = dyn->rs.depth_bias.slope_factor,
      .line_width = dyn->rs.line.width,
      .index_buffer = command->index_buffer,
      .index_offset = command->index_offset,
      .index_type = command->index_type,
   };
   memcpy(draw.vertex_buffers, command->vertex_buffers, sizeof(draw.vertex_buffers));
   memcpy(draw.vertex_offsets, command->vertex_offsets, sizeof(draw.vertex_offsets));
   memcpy(draw.push, command->push, sizeof(draw.push));
   if (!rd_copy_stage_descriptors(command, pipeline->gfx->vs.res, &draw.descriptors[0]) ||
       !rd_copy_stage_descriptors(command, pipeline->gfx->fs.res, &draw.descriptors[1])) {
      vk_free(&command->vk.pool->alloc, draw.descriptors[0]);
      vk_command_buffer_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return NULL;
   }
   util_dynarray_append(&command->render->draws, draw);
   return util_dynarray_top_ptr(&command->render->draws, struct rd_draw);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDraw(VkCommandBuffer handle, uint32_t vertex_count, uint32_t instance_count,
           uint32_t first_vertex, uint32_t first_instance)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (!vertex_count || !instance_count)
      return;
   struct rd_draw *draw = rd_record_draw(command);
   if (!draw)
      return;
   draw->vertex_count = vertex_count;
   draw->instance_count = instance_count;
   draw->first_vertex = first_vertex;
   draw->first_instance = first_instance;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDrawIndexed(VkCommandBuffer handle, uint32_t index_count, uint32_t instance_count,
                  uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (!index_count || !instance_count)
      return;
   if (!command->index_buffer) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   struct rd_draw *draw = rd_record_draw(command);
   if (!draw)
      return;
   draw->indexed = true;
   draw->vertex_count = index_count;
   draw->instance_count = instance_count;
   draw->first_index = first_index;
   draw->vertex_offset = vertex_offset;
   draw->first_instance = first_instance;
}

static void
rd_record_indirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                   uint32_t count, uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   if (indexed && !command->index_buffer) {
      vk_command_buffer_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   for (uint32_t i = 0; i < count; i++) {
      struct rd_draw *draw = rd_record_draw(command);
      if (!draw)
         return;
      draw->indexed = indexed;
      draw->indirect = rd_buffer_from_handle(buffer);
      draw->indirect_offset = offset + (VkDeviceSize)i * stride;
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDrawIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                   uint32_t count, uint32_t stride)
{
   rd_record_indirect(handle, buffer, offset, count, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdDrawIndexedIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                          uint32_t count, uint32_t stride)
{
   rd_record_indirect(handle, buffer, offset, count, stride, true);
}

/* --- emission ------------------------------------------------------------- */

static void
rd_event(struct fd_ringbuffer *ring, struct rd_device *device, enum vgt_event_type evt,
         bool timestamp)
{
   OUT_PKT7(ring, CP_EVENT_WRITE, timestamp ? 4 : 1);
   OUT_RING(ring, CP_EVENT_WRITE_0_EVENT(evt));
   if (timestamp) {
      OUT_RELOC(ring, device->control, 64, 0, 0);
      OUT_RING(ring, 0);
   }
}

static void
rd_gfx_restore(struct fd_ringbuffer *ring, struct rd_device *device)
{
   OUT_PKT7(ring, CP_SET_RENDER_MODE, 5);
   OUT_RING(ring, CP_SET_RENDER_MODE_0_MODE(BYPASS));
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);

   OUT_PKT4(ring, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN, 5);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0x00000012);
   OUT_WFI5(ring);

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0xfffff);
   OUT_PKT4(ring, REG_A5XX_PC_RESTART_INDEX, 1);
   OUT_RING(ring, 0xffffffff);
   OUT_PKT4(ring, REG_A5XX_PC_RASTER_CNTL, 1);
   OUT_RING(ring, 0x00000012);
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
   OUT_RING(ring, A5XX_GRAS_SU_POINT_MINMAX_MIN(1.0f) | A5XX_GRAS_SU_POINT_MINMAX_MAX(4092.0f));
   OUT_RING(ring, A5XX_GRAS_SU_POINT_SIZE(0.5f));
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_CONSERVATIVE_RAS_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_VS_CONFIG_MAX_CONST, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_FS_CONFIG_MAX_CONST, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E292, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_RB_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000044);
   OUT_PKT4(ring, REG_A5XX_RB_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00100000);
   OUT_PKT4(ring, REG_A5XX_VFD_MODE_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_PC_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001f);
   OUT_PKT4(ring, REG_A5XX_SP_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001e);
   OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x40000800);
   OUT_PKT4(ring, REG_A5XX_TPL1_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000544);
   OUT_PKT4(ring, REG_A5XX_HLSQ_TIMEOUT_THRESHOLD_0, 2);
   OUT_RING(ring, 0x00000080);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00000400);
   OUT_PKT4(ring, REG_A5XX_HLSQ_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000001);
   OUT_PKT4(ring, REG_A5XX_VPC_MODE_CNTL, 1);
   OUT_RING(ring, 0);

   OUT_PKT7(ring, CP_SET_DRAW_STATE, 3);
   OUT_RING(ring, CP_SET_DRAW_STATE__0_COUNT(0) | CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS |
                  CP_SET_DRAW_STATE__0_GROUP_ID(0));
   OUT_RING(ring, CP_SET_DRAW_STATE__1_ADDR_LO(0));
   OUT_RING(ring, CP_SET_DRAW_STATE__2_ADDR_HI(0));

   OUT_PKT4(ring, REG_A5XX_GRAS_SC_BIN_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_FS_PRIMITIVEID_CNTL, 1);
   OUT_RING(ring, 0x000000ff);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_OVERRIDE, 1);
   OUT_RING(ring, A5XX_VPC_SO_OVERRIDE_SO_DISABLE);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_BASE(0), 3);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_FLUSH_BASE(0), 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_PC_GS_PARAM, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_PC_HS_PARAM, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_TPL1_TP_FS_ROTATION_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E004, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_LAYERED, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUF_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(0), 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_PC_GS_LAYERED, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5AB, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5C2, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_BASE(1), 3);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(1), 6);
   for (unsigned i = 0; i < 6; i++)
      OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(2), 6);
   for (unsigned i = 0; i < 6; i++)
      OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(3), 3);
   for (unsigned i = 0; i < 3; i++)
      OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5DB, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_HS_CTRL_REG0, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_GS_CTRL_REG0, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_TPL1_VS_TEX_COUNT, 4);
   for (unsigned i = 0; i < 4; i++)
      OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_TPL1_FS_TEX_COUNT, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   static const uint32_t unknown[] = {
      REG_A5XX_UNKNOWN_E7C0, REG_A5XX_UNKNOWN_E7C5, REG_A5XX_UNKNOWN_E7CA,
      REG_A5XX_UNKNOWN_E7CF, REG_A5XX_UNKNOWN_E7D4, REG_A5XX_UNKNOWN_E7D9,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(unknown); i++) {
      OUT_PKT4(ring, unknown[i], 3);
      OUT_RING(ring, 0);
      OUT_RING(ring, 0);
      OUT_RING(ring, 0);
   }
   OUT_PKT4(ring, REG_A5XX_RB_CLEAR_CNTL, 1);
   OUT_RING(ring, 0);
}

/* Depth (D16, D32F, D24S8 with its interleaved stencil) and a stencil-only
 * S8 buffer (as a separate stencil). */
static void
rd_gfx_depth_buffer(struct fd_ringbuffer *ring, const struct rd_render *render)
{
   const struct rd_image_view *dview = render->depth.view ? render->depth.view : render->stencil.view;
   enum a5xx_depth_format fmt = DEPTH5_NONE;
   const struct rd_image *image = NULL;
   const struct rd_image_level *level = NULL;
   VkDeviceSize offset = 0;
   bool separate_stencil = false;
   if (dview) {
      image = container_of(dview->vk.image, struct rd_image, vk);
      level = &image->levels[dview->vk.base_mip_level];
      offset = image->memory_offset + level->offset +
               (VkDeviceSize)dview->vk.base_array_layer * level->array_pitch;
      switch (dview->vk.format) {
      case VK_FORMAT_D16_UNORM: fmt = DEPTH5_16; break;
      case VK_FORMAT_D32_SFLOAT: fmt = DEPTH5_32; break;
      case VK_FORMAT_D24_UNORM_S8_UINT:
      case VK_FORMAT_X8_D24_UNORM_PACK32: fmt = DEPTH5_24_8; break;
      case VK_FORMAT_S8_UINT: separate_stencil = true; break;
      default: break;
      }
   }
   OUT_PKT4(ring, REG_A5XX_RB_DEPTH_BUFFER_INFO, 5);
   OUT_RING(ring, A5XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));
   if (fmt != DEPTH5_NONE) {
      OUT_RELOC(ring, image->memory->bo, offset, 0, 0);
      OUT_RING(ring, A5XX_RB_DEPTH_BUFFER_PITCH(level->row_pitch));
      OUT_RING(ring, A5XX_RB_DEPTH_BUFFER_ARRAY_PITCH(level->array_pitch));
   } else {
      for (unsigned i = 0; i < 4; i++)
         OUT_RING(ring, 0);
   }
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_DEPTH_BUFFER_INFO, 1);
   OUT_RING(ring, A5XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));
   OUT_PKT4(ring, REG_A5XX_RB_DEPTH_FLAG_BUFFER_BASE, 3);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_GRAS_LRZ_BUFFER_BASE, 3);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_GRAS_LRZ_FAST_CLEAR_BUFFER_BASE, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   if (separate_stencil) {
      OUT_PKT4(ring, REG_A5XX_RB_STENCIL_INFO, 5);
      OUT_RING(ring, A5XX_RB_STENCIL_INFO_SEPARATE_STENCIL);
      OUT_RELOC(ring, image->memory->bo, offset, 0, 0);
      OUT_RING(ring, A5XX_RB_STENCIL_PITCH(level->row_pitch));
      OUT_RING(ring, A5XX_RB_STENCIL_ARRAY_PITCH(level->array_pitch));
   } else {
      OUT_PKT4(ring, REG_A5XX_RB_STENCIL_INFO, 1);
      OUT_RING(ring, 0);
   }
}

static void
rd_gfx_sysmem_prep(struct fd_ringbuffer *ring, struct rd_device *device,
                   const struct rd_render *render, uint32_t width, uint32_t height)
{
   OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
   OUT_RING(ring, 0);
   rd_event(ring, device, PC_CCU_INVALIDATE_COLOR, false);
   OUT_PKT4(ring, REG_A5XX_PC_POWER_CNTL, 1);
   OUT_RING(ring, device->info.num_sp_cores - 1);
   OUT_PKT4(ring, REG_A5XX_VFD_POWER_CNTL, 1);
   OUT_RING(ring, device->info.num_sp_cores - 1);
   OUT_WFI5(ring);
   OUT_PKT4(ring, REG_A5XX_RB_CCU_CNTL, 1);
   OUT_RING(ring, 0x10000000);
   OUT_PKT4(ring, REG_A5XX_RB_CNTL, 1);
   OUT_RING(ring, A5XX_RB_CNTL_WIDTH(0) | A5XX_RB_CNTL_HEIGHT(0) | A5XX_RB_CNTL_BYPASS);

   OUT_PKT4(ring, REG_A5XX_GRAS_SC_WINDOW_SCISSOR_TL, 2);
   OUT_RING(ring, A5XX_GRAS_SC_WINDOW_SCISSOR_TL_X(0) | A5XX_GRAS_SC_WINDOW_SCISSOR_TL_Y(0));
   OUT_RING(ring, A5XX_GRAS_SC_WINDOW_SCISSOR_BR_X(width - 1) |
                  A5XX_GRAS_SC_WINDOW_SCISSOR_BR_Y(height - 1));
   OUT_PKT4(ring, REG_A5XX_RB_RESOLVE_CNTL_1, 2);
   OUT_RING(ring, A5XX_RB_RESOLVE_CNTL_1_X(0) | A5XX_RB_RESOLVE_CNTL_1_Y(0));
   OUT_RING(ring, A5XX_RB_RESOLVE_CNTL_2_X(width - 1) | A5XX_RB_RESOLVE_CNTL_2_Y(height - 1));
   OUT_PKT4(ring, REG_A5XX_RB_WINDOW_OFFSET, 1);
   OUT_RING(ring, A5XX_RB_WINDOW_OFFSET_X(0) | A5XX_RB_WINDOW_OFFSET_Y(0));
   OUT_PKT4(ring, REG_A5XX_VPC_SO_OVERRIDE, 1);
   OUT_RING(ring, 0);
   OUT_PKT7(ring, CP_SET_VISIBILITY_OVERRIDE, 1);
   OUT_RING(ring, 0x1);

   rd_gfx_depth_buffer(ring, render);

   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      enum a5xx_color_fmt format = 0;
      enum a3xx_color_swap swap = WZYX;
      bool srgb = false, sint = false, uint = false;
      uint32_t stride = 0, size = 0;
      const struct rd_image *image = NULL;
      VkDeviceSize offset = 0;
      if (i < render->color_count && render->color[i].view) {
         const struct vk_image_view *view = &render->color[i].view->vk;
         image = container_of(view->image, struct rd_image, vk);
         enum pipe_format pformat = vk_format_to_pipe_format(view->format);
         format = rd5_pipe2color(pformat);
         swap = rd5_pipe2swap(pformat);
         srgb = util_format_is_srgb(pformat);
         sint = util_format_is_pure_sint(pformat);
         uint = util_format_is_pure_uint(pformat);
         const struct rd_image_level *level = &image->levels[view->base_mip_level];
         stride = level->row_pitch;
         size = level->array_pitch;
         offset = image->memory_offset + level->offset +
                  (VkDeviceSize)view->base_array_layer * level->array_pitch;
      }
      OUT_PKT4(ring, REG_A5XX_RB_MRT_BUF_INFO(i), 5);
      OUT_RING(ring, A5XX_RB_MRT_BUF_INFO_COLOR_FORMAT(format) |
                     A5XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE(TILE5_LINEAR) |
                     A5XX_RB_MRT_BUF_INFO_COLOR_SWAP(swap) |
                     COND(srgb, A5XX_RB_MRT_BUF_INFO_COLOR_SRGB));
      OUT_RING(ring, A5XX_RB_MRT_PITCH(stride));
      OUT_RING(ring, A5XX_RB_MRT_ARRAY_PITCH(size));
      if (image) {
         OUT_RELOC(ring, image->memory->bo, offset, 0, 0);
      } else {
         OUT_RING(ring, 0);
         OUT_RING(ring, 0);
      }
      OUT_PKT4(ring, REG_A5XX_SP_FS_MRT_REG(i), 1);
      OUT_RING(ring, A5XX_SP_FS_MRT_REG_COLOR_FORMAT(format) |
                     COND(sint, A5XX_SP_FS_MRT_REG_COLOR_SINT) |
                     COND(uint, A5XX_SP_FS_MRT_REG_COLOR_UINT) |
                     COND(srgb, A5XX_SP_FS_MRT_REG_COLOR_SRGB));
      OUT_PKT4(ring, REG_A5XX_RB_MRT_FLAG_BUFFER(i), 4);
      for (unsigned j = 0; j < 4; j++)
         OUT_RING(ring, 0);
   }

   enum a3xx_msaa_samples samples = MSAA_ONE;
   OUT_PKT4(ring, REG_A5XX_TPL1_TP_RAS_MSAA_CNTL, 2);
   OUT_RING(ring, A5XX_TPL1_TP_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(ring, A5XX_TPL1_TP_DEST_MSAA_CNTL_SAMPLES(samples) |
                  A5XX_TPL1_TP_DEST_MSAA_CNTL_MSAA_DISABLE);
   OUT_PKT4(ring, REG_A5XX_RB_RAS_MSAA_CNTL, 2);
   OUT_RING(ring, A5XX_RB_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(ring, A5XX_RB_DEST_MSAA_CNTL_SAMPLES(samples) | A5XX_RB_DEST_MSAA_CNTL_MSAA_DISABLE);
   OUT_PKT4(ring, REG_A5XX_GRAS_SC_RAS_MSAA_CNTL, 2);
   OUT_RING(ring, A5XX_GRAS_SC_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(ring, A5XX_GRAS_SC_DEST_MSAA_CNTL_SAMPLES(samples) |
                  A5XX_GRAS_SC_DEST_MSAA_CNTL_MSAA_DISABLE);
}

static void
rd_gfx_shader_obj(struct fd_ringbuffer *ring, const struct rd_gfx_shader *s, uint32_t reg)
{
   OUT_PKT4(ring, reg, 6);
   OUT_RELOC(ring, s->bo, 0, 0, 0);
   OUT_RING(ring, 0); /* PVT_MEM_PARAM: no private memory */
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
}

static void
rd_gfx_load_shader(struct fd_ringbuffer *ring, const struct rd_gfx_shader *s,
                   enum a4xx_state_block sb, unsigned instrlen)
{
   OUT_PKT7(ring, CP_LOAD_STATE4, 3);
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(0) | CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                  CP_LOAD_STATE4_0_STATE_BLOCK(sb) | CP_LOAD_STATE4_0_NUM_UNIT(instrlen));
   OUT_RELOC(ring, s->bo, 0, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER), 0);
}

static inline uint32_t
rd_next_regid(uint32_t reg, uint32_t increment)
{
   return VALIDREG(reg) ? reg + increment : regid(63, 0);
}

/* fd5_program_emit for VS + FS. Returns the VPC stride (max_loc). */
static unsigned
rd_gfx_program(struct fd_ringbuffer *ring, const struct rd_gfx_pipeline *gfx)
{
   const struct ir3_shader_variant *vs = gfx->vs.v, *fs = gfx->fs.v;
   unsigned vs_constlen = vs->constlen / 4, fs_constlen = fs->constlen / 4;
   unsigned vs_instrlen = vs->instrlen, fs_instrlen = fs->instrlen;
   if (vs_instrlen + fs_instrlen > 64) {
      if (fs_instrlen < 64)
         vs_instrlen = 0;
      else if (vs_instrlen < 64)
         fs_instrlen = 0;
      else
         vs_instrlen = fs_instrlen = 0;
   }
   unsigned vs_constoff = 0, fs_constoff = vs_constlen;
   unsigned vs_instroff = 0, fs_instroff = 64 - fs_instrlen;
   enum a3xx_threadsize fssz = fs->info.double_threadsize ? FOUR_QUADS : TWO_QUADS;

   uint32_t pos_regid = ir3_find_output_regid(vs, VARYING_SLOT_POS);
   uint32_t psize_regid = ir3_find_output_regid(vs, VARYING_SLOT_PSIZ);
   uint32_t vertex_regid = ir3_find_sysval_regid(vs, SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   uint32_t instance_regid = ir3_find_sysval_regid(vs, SYSTEM_VALUE_INSTANCE_ID);
   uint32_t color_regid[8];
   for (unsigned i = 0; i < 8; i++)
      color_regid[i] = fs->color0_mrt ? ir3_find_output_regid(fs, FRAG_RESULT_COLOR) :
                                        ir3_find_output_regid(fs, FRAG_RESULT_DATA0 + i);
   uint32_t samp_id_regid = ir3_find_sysval_regid(fs, SYSTEM_VALUE_SAMPLE_ID);
   uint32_t samp_mask_regid = ir3_find_sysval_regid(fs, SYSTEM_VALUE_SAMPLE_MASK_IN);
   uint32_t face_regid = ir3_find_sysval_regid(fs, SYSTEM_VALUE_FRONT_FACE);
   uint32_t coord_regid = ir3_find_sysval_regid(fs, SYSTEM_VALUE_FRAG_COORD);
   uint32_t zwcoord_regid = rd_next_regid(coord_regid, 2);
   uint32_t ij_regid[IJ_COUNT];
   for (unsigned i = 0; i < IJ_COUNT; i++)
      ij_regid[i] = ir3_find_sysval_regid(fs, SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL + i);

   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CONFIG, 5);
   OUT_RING(ring, A5XX_HLSQ_VS_CONFIG_CONSTOBJECTOFFSET(vs_constoff) |
                  A5XX_HLSQ_VS_CONFIG_SHADEROBJOFFSET(vs_instroff) | A5XX_HLSQ_VS_CONFIG_ENABLED);
   OUT_RING(ring, A5XX_HLSQ_FS_CONFIG_CONSTOBJECTOFFSET(fs_constoff) |
                  A5XX_HLSQ_FS_CONFIG_SHADEROBJOFFSET(fs_instroff) | A5XX_HLSQ_FS_CONFIG_ENABLED);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONFIG, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CNTL, 5);
   OUT_RING(ring, A5XX_HLSQ_VS_CNTL_INSTRLEN(vs_instrlen));
   OUT_RING(ring, A5XX_HLSQ_FS_CNTL_INSTRLEN(fs_instrlen));
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_VS_CONFIG, 5);
   OUT_RING(ring, A5XX_SP_VS_CONFIG_CONSTOBJECTOFFSET(vs_constoff) |
                  A5XX_SP_VS_CONFIG_SHADEROBJOFFSET(vs_instroff) | A5XX_SP_VS_CONFIG_ENABLED);
   OUT_RING(ring, A5XX_SP_FS_CONFIG_CONSTOBJECTOFFSET(fs_constoff) |
                  A5XX_SP_FS_CONFIG_SHADEROBJOFFSET(fs_instroff) | A5XX_SP_FS_CONFIG_ENABLED);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CONSTLEN, 2);
   OUT_RING(ring, vs_constlen);
   OUT_RING(ring, vs_instrlen);
   OUT_PKT4(ring, REG_A5XX_HLSQ_FS_CONSTLEN, 2);
   OUT_RING(ring, fs_constlen);
   OUT_RING(ring, fs_instrlen);
   OUT_PKT4(ring, REG_A5XX_HLSQ_HS_CONSTLEN, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_DS_CONSTLEN, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_GS_CONSTLEN, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(ring, 0);
   OUT_RING(ring, 0);

   OUT_PKT4(ring, REG_A5XX_SP_VS_CTRL_REG0, 1);
   OUT_RING(ring, A5XX_SP_VS_CTRL_REG0_HALFREGFOOTPRINT(vs->info.max_half_reg + 1) |
                  A5XX_SP_VS_CTRL_REG0_FULLREGFOOTPRINT(vs->info.max_reg + 1) |
                  COND(vs_instrlen != 0, A5XX_SP_VS_CTRL_REG0_BUFFER) | 0x2 |
                  A5XX_SP_VS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(vs)) |
                  COND(vs->need_pixlod, A5XX_SP_VS_CTRL_REG0_PIXLODENABLE));

   struct ir3_shader_linkage l = {0};
   ir3_link_shaders(&l, vs, fs, true);
   OUT_PKT4(ring, REG_A5XX_VPC_VAR_DISABLE(0), 4);
   for (unsigned i = 0; i < 4; i++)
      OUT_RING(ring, ~l.varmask[i]);
   uint8_t pos_loc = l.max_loc, psize_loc = ~0;
   if (VALIDREG(pos_regid))
      ir3_link_add(&l, VARYING_SLOT_POS, pos_regid, 0xf, l.max_loc);
   if (VALIDREG(psize_regid)) {
      psize_loc = l.max_loc;
      ir3_link_add(&l, VARYING_SLOT_PSIZ, psize_regid, 0x1, l.max_loc);
   }
   for (unsigned i = 0, j = 0; i < 16 && j < l.cnt; i++) {
      uint32_t reg = A5XX_SP_VS_OUT_REG_A_REGID(l.var[j].regid) |
                     A5XX_SP_VS_OUT_REG_A_COMPMASK(l.var[j].compmask);
      j++;
      reg |= A5XX_SP_VS_OUT_REG_B_REGID(l.var[j].regid) |
             A5XX_SP_VS_OUT_REG_B_COMPMASK(l.var[j].compmask);
      j++;
      OUT_PKT4(ring, REG_A5XX_SP_VS_OUT_REG(i), 1);
      OUT_RING(ring, reg);
   }
   for (unsigned i = 0, j = 0; i < 8 && j < l.cnt; i++) {
      uint32_t reg = A5XX_SP_VS_VPC_DST_REG_OUTLOC0(l.var[j].loc) |
                     A5XX_SP_VS_VPC_DST_REG_OUTLOC1(l.var[j + 1].loc) |
                     A5XX_SP_VS_VPC_DST_REG_OUTLOC2(l.var[j + 2].loc) |
                     A5XX_SP_VS_VPC_DST_REG_OUTLOC3(l.var[j + 3].loc);
      j += 4;
      OUT_PKT4(ring, REG_A5XX_SP_VS_VPC_DST_REG(i), 1);
      OUT_RING(ring, reg);
   }
   rd_gfx_shader_obj(ring, &gfx->vs, REG_A5XX_SP_VS_OBJ_START);
   if (vs_instrlen)
      rd_gfx_load_shader(ring, &gfx->vs, SB4_VS_SHADER, vs_instrlen);

   OUT_PKT4(ring, REG_A5XX_PC_PRIM_VTX_CNTL, 1);
   OUT_RING(ring, COND(vs->writes_psize, A5XX_PC_PRIM_VTX_CNTL_PSIZE));
   OUT_PKT4(ring, REG_A5XX_SP_PRIMITIVE_CNTL, 1);
   OUT_RING(ring, A5XX_SP_PRIMITIVE_CNTL_VSOUT(l.cnt));
   OUT_PKT4(ring, REG_A5XX_VPC_CNTL_0, 1);
   OUT_RING(ring, A5XX_VPC_CNTL_0_STRIDE_IN_VPC(l.max_loc) |
                  COND(fs->total_in > 0, A5XX_VPC_CNTL_0_VARYING) | 0x10000);

   rd_gfx_shader_obj(ring, &gfx->fs, REG_A5XX_SP_FS_OBJ_START);
   OUT_PKT4(ring, REG_A5XX_HLSQ_CONTROL_0_REG, 5);
   OUT_RING(ring, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(fssz) |
                  A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(TWO_QUADS) | 0x00000880);
   OUT_RING(ring, A5XX_HLSQ_CONTROL_1_REG_PRIMALLOCTHRESHOLD(63));
   OUT_RING(ring, A5XX_HLSQ_CONTROL_2_REG_FACEREGID(face_regid) |
                  A5XX_HLSQ_CONTROL_2_REG_SAMPLEID(samp_id_regid) |
                  A5XX_HLSQ_CONTROL_2_REG_SAMPLEMASK(samp_mask_regid) |
                  A5XX_HLSQ_CONTROL_2_REG_CENTERRHW(ij_regid[IJ_PERSP_CENTER_RHW]));
   OUT_RING(ring, A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_PIXEL(ij_regid[IJ_PERSP_PIXEL]) |
                  A5XX_HLSQ_CONTROL_3_REG_IJ_LINEAR_PIXEL(ij_regid[IJ_LINEAR_PIXEL]) |
                  A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_CENTROID(ij_regid[IJ_PERSP_CENTROID]) |
                  A5XX_HLSQ_CONTROL_3_REG_IJ_LINEAR_CENTROID(ij_regid[IJ_LINEAR_CENTROID]));
   OUT_RING(ring, A5XX_HLSQ_CONTROL_4_REG_XYCOORDREGID(coord_regid) |
                  A5XX_HLSQ_CONTROL_4_REG_ZWCOORDREGID(zwcoord_regid) |
                  A5XX_HLSQ_CONTROL_4_REG_IJ_PERSP_SAMPLE(ij_regid[IJ_PERSP_SAMPLE]) |
                  A5XX_HLSQ_CONTROL_4_REG_IJ_LINEAR_SAMPLE(ij_regid[IJ_LINEAR_SAMPLE]));
   OUT_PKT4(ring, REG_A5XX_SP_FS_CTRL_REG0, 1);
   OUT_RING(ring, COND(fs->total_in > 0, A5XX_SP_FS_CTRL_REG0_VARYING) | 0x40002 |
                  COND(fs_instrlen != 0, A5XX_SP_FS_CTRL_REG0_BUFFER) |
                  A5XX_SP_FS_CTRL_REG0_THREADSIZE(fssz) |
                  A5XX_SP_FS_CTRL_REG0_HALFREGFOOTPRINT(fs->info.max_half_reg + 1) |
                  A5XX_SP_FS_CTRL_REG0_FULLREGFOOTPRINT(fs->info.max_reg + 1) |
                  A5XX_SP_FS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(fs)) |
                  COND(fs->need_pixlod, A5XX_SP_FS_CTRL_REG0_PIXLODENABLE));
   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x020fffff);
   OUT_PKT4(ring, REG_A5XX_VPC_GS_SIV_CNTL, 1);
   OUT_RING(ring, 0x0000ffff);
   OUT_PKT4(ring, REG_A5XX_SP_SP_CNTL, 1);
   OUT_RING(ring, 0x00000010);

   uint32_t ij_bits = 0, rb_ij_bits = 0;
   static const uint32_t gras_ij[IJ_COUNT] = {
      [IJ_PERSP_PIXEL] = A5XX_GRAS_CNTL_IJ_PERSP_PIXEL,
      [IJ_PERSP_SAMPLE] = A5XX_GRAS_CNTL_IJ_PERSP_SAMPLE,
      [IJ_PERSP_CENTROID] = A5XX_GRAS_CNTL_IJ_PERSP_CENTROID,
      [IJ_PERSP_CENTER_RHW] = A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL,
      [IJ_LINEAR_PIXEL] = A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL,
      [IJ_LINEAR_CENTROID] = A5XX_GRAS_CNTL_IJ_LINEAR_CENTROID,
      [IJ_LINEAR_SAMPLE] = A5XX_GRAS_CNTL_IJ_LINEAR_SAMPLE,
   };
   static const uint32_t rb_ij[IJ_COUNT] = {
      [IJ_PERSP_PIXEL] = A5XX_RB_RENDER_CONTROL0_IJ_PERSP_PIXEL,
      [IJ_PERSP_SAMPLE] = A5XX_RB_RENDER_CONTROL0_IJ_PERSP_SAMPLE,
      [IJ_PERSP_CENTROID] = A5XX_RB_RENDER_CONTROL0_IJ_PERSP_CENTROID,
      [IJ_PERSP_CENTER_RHW] = A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL,
      [IJ_LINEAR_PIXEL] = A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL,
      [IJ_LINEAR_CENTROID] = A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_CENTROID,
      [IJ_LINEAR_SAMPLE] = A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_SAMPLE,
   };
   for (unsigned i = 0; i < IJ_COUNT; i++) {
      if (VALIDREG(ij_regid[i])) {
         ij_bits |= gras_ij[i];
         rb_ij_bits |= rb_ij[i];
      }
   }
   if (fs->fragcoord_compmask) {
      ij_bits |= A5XX_GRAS_CNTL_COORD_MASK(fs->fragcoord_compmask) | A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL;
      rb_ij_bits |= A5XX_RB_RENDER_CONTROL0_COORD_MASK(fs->fragcoord_compmask) |
                    A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL;
   }
   if (fs->frag_face) {
      ij_bits |= A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL;
      rb_ij_bits |= A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL;
   }
   OUT_PKT4(ring, REG_A5XX_GRAS_CNTL, 1);
   OUT_RING(ring, ij_bits);
   OUT_PKT4(ring, REG_A5XX_RB_RENDER_CONTROL0, 2);
   OUT_RING(ring, rb_ij_bits);
   OUT_RING(ring, COND(VALIDREG(samp_mask_regid), A5XX_RB_RENDER_CONTROL1_SAMPLEMASK) |
                  COND(fs->frag_face, A5XX_RB_RENDER_CONTROL1_FACENESS) |
                  COND(VALIDREG(samp_id_regid), A5XX_RB_RENDER_CONTROL1_SAMPLEID) |
                  COND(VALIDREG(ij_regid[IJ_PERSP_CENTER_RHW]), A5XX_RB_RENDER_CONTROL1_CENTERRHW) |
                  COND(fs->sample_shading, A5XX_RB_RENDER_CONTROL1_SAMPLEMODE(3)));
   OUT_PKT4(ring, REG_A5XX_SP_FS_OUTPUT_REG(0), 8);
   for (unsigned i = 0; i < 8; i++)
      OUT_RING(ring, A5XX_SP_FS_OUTPUT_REG_REGID(color_regid[i]) |
                     COND(color_regid[i] & HALF_REG_ID, A5XX_SP_FS_OUTPUT_REG_HALF_PRECISION));
   OUT_PKT4(ring, REG_A5XX_VPC_PACK, 1);
   OUT_RING(ring, A5XX_VPC_PACK_NUMNONPOSVAR(pos_loc) | A5XX_VPC_PACK_PSIZELOC(psize_loc));

   uint32_t vinterp[8] = {0}, vpsrepl[8] = {0};
   for (int j = -1; (j = ir3_next_varying(fs, j)) < (int)fs->inputs_count;) {
      unsigned compmask = fs->inputs[j].compmask;
      uint32_t loc = fs->inputs[j].inloc;
      if (fs->inputs[j].flat) {
         for (unsigned i = 0; i < 4; i++) {
            if (compmask & (1 << i)) {
               vinterp[loc / 16] |= 1 << ((loc % 16) * 2);
               loc++;
            }
         }
      }
   }
   OUT_PKT4(ring, REG_A5XX_VPC_VARYING_INTERP_MODE(0), 8);
   for (unsigned i = 0; i < 8; i++)
      OUT_RING(ring, vinterp[i]);
   OUT_PKT4(ring, REG_A5XX_VPC_VARYING_PS_REPL_MODE(0), 8);
   for (unsigned i = 0; i < 8; i++)
      OUT_RING(ring, vpsrepl[i]);
   OUT_PKT4(ring, REG_A5XX_GRAS_VS_CL_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_VPC_CLIP_CNTL, 1);
   OUT_RING(ring, A5XX_VPC_CLIP_CNTL_CLIP_DIST_03_LOC(0xff) |
                  A5XX_VPC_CLIP_CNTL_CLIP_DIST_47_LOC(0xff));
   OUT_PKT4(ring, REG_A5XX_PC_CLIP_CNTL, 1);
   OUT_RING(ring, 0);
   if (fs_instrlen)
      rd_gfx_load_shader(ring, &gfx->fs, SB4_FS_SHADER, fs_instrlen);
   OUT_PKT4(ring, REG_A5XX_VFD_CONTROL_1, 5);
   OUT_RING(ring, A5XX_VFD_CONTROL_1_REGID4VTX(vertex_regid) |
                  A5XX_VFD_CONTROL_1_REGID4INST(instance_regid) | 0xfc0000);
   OUT_RING(ring, 0x0000fcfc);
   OUT_RING(ring, 0x0000fcfc);
   OUT_RING(ring, 0x000000fc);
   OUT_RING(ring, 0);
   return l.max_loc;
}

static void
rd_gfx_const(struct fd_ringbuffer *ring, enum a4xx_state_block sb, uint32_t constid,
             uint32_t sizedwords, const uint32_t *dwords)
{
   uint32_t align_sz = align(sizedwords, 4);
   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + align_sz);
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(constid / 4) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                  CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                  CP_LOAD_STATE4_0_NUM_UNIT(DIV_ROUND_UP(sizedwords, 4)));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
   OUT_RING(ring, 0);
   for (unsigned i = 0; i < sizedwords; i++)
      OUT_RING(ring, dwords[i]);
   for (unsigned i = sizedwords; i < align_sz; i++)
      OUT_RING(ring, 0);
}

static void
rd_gfx_consts(struct fd_ringbuffer *ring, struct rd_device *device,
              const struct rd_gfx_shader *shader, enum a4xx_state_block sb,
              const struct rd_draw *draw, const struct rd_descriptor *descriptors,
              int32_t vertex_base)
{
   const struct ir3_shader_variant *v = shader->v;
   const struct ir3_const_state *state = ir3_const_state(v);
   unsigned constlen = v->constlen * 4;
   /* push constants */
   rd_gfx_const(ring, sb, 0, MIN2(RD_GFX_PUSH_DWORDS, constlen), draw->push);
   /* buffer addresses, two dwords each */
   const struct rd_pipeline *res = shader->res;
   if (res->buffer_count && RD_GFX_PUSH_DWORDS < constlen) {
      unsigned count = res->buffer_count;
      unsigned dwords = align(2 * count, 4);
      OUT_PKT7(ring, CP_LOAD_STATE4, 3 + dwords);
      OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(RD_GFX_PUSH_DWORDS / 4) |
                     CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) | CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                     CP_LOAD_STATE4_0_NUM_UNIT(dwords / 4));
      OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
      OUT_RING(ring, 0);
      for (unsigned i = 0; i < count; i++) {
         const struct rd_descriptor *d = &descriptors[res->buffer_descriptors[i]];
         if (d->buffer && d->buffer->memory)
            OUT_RELOC(ring, d->buffer->memory->bo, d->buffer->offset + d->offset, 0, 0);
         else
            OUT_RELOC(ring, device->control, 0, 0, 0);
      }
      for (unsigned i = 2 * count; i < dwords; i++)
         OUT_RING(ring, 0);
   }
   /* constant data (e.g. dynamically indexed constant arrays) */
   if (v->constant_data_size) {
      if (ir3_const_can_upload(&state->allocs, IR3_CONST_ALLOC_UBO_PTRS, v->constlen)) {
         unsigned base = state->allocs.consts[IR3_CONST_ALLOC_UBO_PTRS].offset_vec4;
         unsigned count = align(state->num_ubos, 2);
         OUT_PKT7(ring, CP_LOAD_STATE4, 3 + 2 * count);
         OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(base) | CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                        CP_LOAD_STATE4_0_STATE_BLOCK(sb) | CP_LOAD_STATE4_0_NUM_UNIT(count / 2));
         OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
         OUT_RING(ring, 0);
         for (unsigned i = 0; i < count; i++) {
            if (i == (unsigned)state->consts_ubo.idx) {
               OUT_RELOC(ring, shader->bo, v->info.constant_data_offset, 0, 0);
            } else {
               OUT_RING(ring, 0);
               OUT_RING(ring, 0);
            }
         }
      }
      for (unsigned i = 0; i < state->ubo_state.num_enabled; i++) {
         const struct ir3_ubo_range *range = &state->ubo_state.range[i];
         if (range->ubo.block != (uint32_t)state->consts_ubo.idx || range->offset >= 16 * v->constlen)
            continue;
         unsigned size = MIN2(range->end - range->start, 16 * v->constlen - range->offset);
         if (size)
            rd_gfx_const(ring, sb, range->offset / 4, size / 4,
                         (const uint32_t *)((const uint8_t *)v->bin +
                                            v->info.constant_data_offset + range->start));
      }
   }
   unsigned base = state->allocs.max_const_offset_vec4 * 4;
   if (v->imm_state.count && base < constlen)
      rd_gfx_const(ring, sb, base, MIN2(align(v->imm_state.count, 4), constlen - base),
                   v->imm_state.values);
   if (sb == SB4_VS_SHADER && v->need_driver_params) {
      struct ir3_driver_params_vs params = {
         .vtxid_base = vertex_base,
         .instid_base = draw->first_instance,
         .is_indexed_draw = draw->indexed ? ~0u : 0,
      };
      unsigned dp = state->allocs.consts[IR3_CONST_ALLOC_DRIVER_PARAMS].offset_vec4 * 4;
      if (dp < constlen) {
         unsigned size = MIN2(state->num_driver_params, constlen - dp);
         if (size)
            rd_gfx_const(ring, sb, dp, size, (const uint32_t *)&params);
      }
   }
}

static void
rd_gfx_vertex_buffers(struct fd_ringbuffer *ring, struct rd_device *device,
                      const struct rd_gfx_pipeline *gfx, const struct rd_draw *draw)
{
   const struct ir3_shader_variant *vs = gfx->vs.v;
   const struct vk_vertex_input_state *vi = gfx->state.vi;
   unsigned j = 0;
   for (unsigned i = 0; i < vs->inputs_count; i++) {
      if (vs->inputs[i].sysval || !vs->inputs[i].compmask)
         continue;
      unsigned location = vs->inputs[i].slot - VERT_ATTRIB_GENERIC0;
      if (!vi || location >= MESA_VK_MAX_VERTEX_ATTRIBUTES ||
          !(vi->attributes_valid & (1u << location)))
         continue;
      const struct vk_vertex_attribute_state *attr = &vi->attributes[location];
      const struct vk_vertex_binding_state *binding = &vi->bindings[attr->binding];
      enum pipe_format pfmt = vk_format_to_pipe_format(attr->format);
      const struct rd_buffer *buffer = draw->vertex_buffers[attr->binding];
      VkDeviceSize offset = draw->vertex_offsets[attr->binding] + attr->offset;
      uint32_t size = 0;
      OUT_PKT4(ring, REG_A5XX_VFD_FETCH(j), 4);
      if (buffer && buffer->memory && offset < buffer->vk.size) {
         size = MIN2(buffer->vk.size - offset, UINT32_MAX);
         OUT_RELOC(ring, buffer->memory->bo, buffer->offset + offset, 0, 0);
      } else {
         OUT_RELOC(ring, device->control, 0, 0, 0);
      }
      OUT_RING(ring, size);
      OUT_RING(ring, binding->stride);
      OUT_PKT4(ring, REG_A5XX_VFD_DECODE(j), 2);
      OUT_RING(ring, A5XX_VFD_DECODE_INSTR_IDX(j) | A5XX_VFD_DECODE_INSTR_FORMAT(rd5_pipe2vtx(pfmt)) |
                     COND(binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
                          A5XX_VFD_DECODE_INSTR_INSTANCED) |
                     A5XX_VFD_DECODE_INSTR_SWAP(rd5_pipe2swap(pfmt)) | A5XX_VFD_DECODE_INSTR_UNK30 |
                     COND(!util_format_is_pure_integer(pfmt), A5XX_VFD_DECODE_INSTR_FLOAT));
      OUT_RING(ring, MAX2(1, binding->divisor));
      OUT_PKT4(ring, REG_A5XX_VFD_DEST_CNTL(j), 1);
      OUT_RING(ring, A5XX_VFD_DEST_CNTL_INSTR_WRITEMASK(vs->inputs[i].compmask) |
                     A5XX_VFD_DEST_CNTL_INSTR_REGID(vs->inputs[i].regid));
      j++;
   }
   OUT_PKT4(ring, REG_A5XX_VFD_CONTROL_0, 1);
   OUT_RING(ring, A5XX_VFD_CONTROL_0_VTXCNT(j));
}

static enum adreno_rb_blend_factor
rd_blend_factor(VkBlendFactor f)
{
   switch (f) {
   case VK_BLEND_FACTOR_ZERO: return FACTOR_ZERO;
   case VK_BLEND_FACTOR_ONE: return FACTOR_ONE;
   case VK_BLEND_FACTOR_SRC_COLOR: return FACTOR_SRC_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return FACTOR_ONE_MINUS_SRC_COLOR;
   case VK_BLEND_FACTOR_DST_COLOR: return FACTOR_DST_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return FACTOR_ONE_MINUS_DST_COLOR;
   case VK_BLEND_FACTOR_SRC_ALPHA: return FACTOR_SRC_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return FACTOR_ONE_MINUS_SRC_ALPHA;
   case VK_BLEND_FACTOR_DST_ALPHA: return FACTOR_DST_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return FACTOR_ONE_MINUS_DST_ALPHA;
   case VK_BLEND_FACTOR_CONSTANT_COLOR: return FACTOR_CONSTANT_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return FACTOR_ONE_MINUS_CONSTANT_COLOR;
   case VK_BLEND_FACTOR_CONSTANT_ALPHA: return FACTOR_CONSTANT_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return FACTOR_ONE_MINUS_CONSTANT_ALPHA;
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return FACTOR_SRC_ALPHA_SATURATE;
   case VK_BLEND_FACTOR_SRC1_COLOR: return FACTOR_SRC1_COLOR;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return FACTOR_ONE_MINUS_SRC1_COLOR;
   case VK_BLEND_FACTOR_SRC1_ALPHA: return FACTOR_SRC1_ALPHA;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return FACTOR_ONE_MINUS_SRC1_ALPHA;
   default: return FACTOR_ZERO;
   }
}

static enum a3xx_rb_blend_opcode
rd_blend_op(VkBlendOp op)
{
   switch (op) {
   case VK_BLEND_OP_ADD: return BLEND_DST_PLUS_SRC;
   case VK_BLEND_OP_SUBTRACT: return BLEND_SRC_MINUS_DST;
   case VK_BLEND_OP_REVERSE_SUBTRACT: return BLEND_DST_MINUS_SRC;
   case VK_BLEND_OP_MIN: return BLEND_MIN_DST_SRC;
   case VK_BLEND_OP_MAX: return BLEND_MAX_DST_SRC;
   default: return BLEND_DST_PLUS_SRC;
   }
}

static enum pc_di_primtype
rd_primtype(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return DI_PT_POINTLIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return DI_PT_LINELIST;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return DI_PT_LINESTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return DI_PT_TRILIST;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: return DI_PT_TRISTRIP;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return DI_PT_TRIFAN;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: return DI_PT_LINE_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return DI_PT_LINESTRIP_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: return DI_PT_TRI_ADJ;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: return DI_PT_TRISTRIP_ADJ;
   default: return DI_PT_TRILIST;
   }
}

static void
rd_gfx_draw(struct fd_ringbuffer *ring, struct rd_device *device,
            const struct rd_render *render, const struct rd_draw *draw)
{
   const struct rd_gfx_pipeline *gfx = draw->pipeline->gfx;
   const struct vk_dynamic_graphics_state *dyn = &gfx->dynamic;
   unsigned nr = 0;
   uint32_t components = 0;
   for (unsigned i = 0; i < render->color_count; i++) {
      if (render->color[i].view) {
         components |= 0xf << (4 * i);
         nr = i + 1;
      }
   }
   OUT_PKT4(ring, REG_A5XX_RB_RENDER_COMPONENTS, 1);
   OUT_RING(ring, components);
   /* depth and stencil (VkCompareOp and VkStencilOp map 1:1) */
   const struct vk_depth_stencil_state *ds = &draw->ds;
   bool has_depth = render->depth.view && vk_format_has_depth(render->depth.view->vk.format);
   const struct rd_image_view *sview = render->stencil.view;
   bool has_stencil = sview && vk_format_has_stencil(sview->vk.format);
   uint32_t depth_cntl = 0, stencil_cntl = 0, refmask = 0, refmask_bf = 0;
   if (has_depth && ds->depth.test_enable)
      depth_cntl |= A5XX_RB_DEPTH_CNTL_ZFUNC((enum adreno_compare_func)ds->depth.compare_op) |
                    A5XX_RB_DEPTH_CNTL_Z_TEST_ENABLE | A5XX_RB_DEPTH_CNTL_Z_READ_ENABLE |
                    COND(ds->depth.write_enable, A5XX_RB_DEPTH_CNTL_Z_WRITE_ENABLE);
   if (has_stencil && ds->stencil.test_enable) {
      const struct vk_stencil_test_face_state *f = &ds->stencil.front, *b = &ds->stencil.back;
      stencil_cntl = A5XX_RB_STENCIL_CONTROL_STENCIL_READ | A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE |
                     A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE_BF |
                     A5XX_RB_STENCIL_CONTROL_FUNC((enum adreno_compare_func)f->op.compare) | A5XX_RB_STENCIL_CONTROL_FAIL((enum adreno_stencil_op)f->op.fail) |
                     A5XX_RB_STENCIL_CONTROL_ZPASS((enum adreno_stencil_op)f->op.pass) | A5XX_RB_STENCIL_CONTROL_ZFAIL((enum adreno_stencil_op)f->op.depth_fail) |
                     A5XX_RB_STENCIL_CONTROL_FUNC_BF((enum adreno_compare_func)b->op.compare) | A5XX_RB_STENCIL_CONTROL_FAIL_BF((enum adreno_stencil_op)b->op.fail) |
                     A5XX_RB_STENCIL_CONTROL_ZPASS_BF((enum adreno_stencil_op)b->op.pass) |
                     A5XX_RB_STENCIL_CONTROL_ZFAIL_BF((enum adreno_stencil_op)b->op.depth_fail);
      refmask = A5XX_RB_STENCILREFMASK_STENCILREF(f->reference) |
                A5XX_RB_STENCILREFMASK_STENCILMASK(f->compare_mask) |
                A5XX_RB_STENCILREFMASK_STENCILWRITEMASK(f->write_mask);
      refmask_bf = A5XX_RB_STENCILREFMASK_BF_STENCILREF(b->reference) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILMASK(b->compare_mask) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILWRITEMASK(b->write_mask);
   }
   bool fragz = gfx->fs.v->no_earlyz || gfx->fs.v->has_kill || gfx->fs.v->writes_pos;
   OUT_PKT4(ring, REG_A5XX_RB_ALPHA_CONTROL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_RB_STENCIL_CONTROL, 1);
   OUT_RING(ring, stencil_cntl);
   OUT_PKT4(ring, REG_A5XX_RB_STENCILREFMASK, 2);
   OUT_RING(ring, refmask);
   OUT_RING(ring, refmask_bf);
   OUT_PKT4(ring, REG_A5XX_RB_DEPTH_CNTL, 1);
   OUT_RING(ring, depth_cntl);
   OUT_PKT4(ring, REG_A5XX_GRAS_LRZ_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_RB_DEPTH_PLANE_CNTL, 1);
   OUT_RING(ring, COND(fragz, A5XX_RB_DEPTH_PLANE_CNTL_FRAG_WRITES_Z) |
                  COND(fragz && gfx->fs.v->fragcoord_compmask, A5XX_RB_DEPTH_PLANE_CNTL_UNK1));
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_DEPTH_PLANE_CNTL, 1);
   OUT_RING(ring, COND(fragz, A5XX_GRAS_SU_DEPTH_PLANE_CNTL_FRAG_WRITES_Z) |
                  COND(fragz && gfx->fs.v->fragcoord_compmask, A5XX_GRAS_SU_DEPTH_PLANE_CNTL_UNK1));

   /* Scissor: the dynamic or static scissor clipped to the render area. */
   int32_t minx = MAX2(draw->scissor.offset.x, render->area.offset.x);
   int32_t miny = MAX2(draw->scissor.offset.y, render->area.offset.y);
   int32_t maxx = MIN2((int64_t)draw->scissor.offset.x + draw->scissor.extent.width,
                       (int64_t)render->area.offset.x + render->area.extent.width) - 1;
   int32_t maxy = MIN2((int64_t)draw->scissor.offset.y + draw->scissor.extent.height,
                       (int64_t)render->area.offset.y + render->area.extent.height) - 1;
   if (maxx < minx || maxy < miny) {
      /* Empty scissor: nothing to draw. */
      return;
   }
   OUT_PKT4(ring, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0, 2);
   OUT_RING(ring, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(minx) | A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(miny));
   OUT_RING(ring, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(maxx) | A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(maxy));
   OUT_PKT4(ring, REG_A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0, 2);
   OUT_RING(ring, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(minx) | A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(miny));
   OUT_RING(ring, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(maxx) | A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(maxy));

   const VkViewport *vp = &draw->viewport;
   OUT_WFI5(ring);
   OUT_PKT4(ring, REG_A5XX_GRAS_CL_VPORT_XOFFSET_0, 6);
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_XOFFSET_0(vp->x + vp->width * 0.5f));
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_XSCALE_0(vp->width * 0.5f));
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_YOFFSET_0(vp->y + vp->height * 0.5f));
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_YSCALE_0(vp->height * 0.5f));
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_ZOFFSET_0(vp->minDepth));
   OUT_RING(ring, A5XX_GRAS_CL_VPORT_ZSCALE_0(vp->maxDepth - vp->minDepth));

   unsigned max_loc = rd_gfx_program(ring, gfx);

   uint32_t su_cntl = A5XX_GRAS_SU_CNTL_LINEHALFWIDTH(draw->line_width / 2.0f) |
                      COND(draw->bias_enable, A5XX_GRAS_SU_CNTL_POLY_OFFSET);
   if (draw->cull & VK_CULL_MODE_FRONT_BIT)
      su_cntl |= A5XX_GRAS_SU_CNTL_CULL_FRONT;
   if (draw->cull & VK_CULL_MODE_BACK_BIT)
      su_cntl |= A5XX_GRAS_SU_CNTL_CULL_BACK;
   if (draw->front == VK_FRONT_FACE_CLOCKWISE)
      su_cntl |= A5XX_GRAS_SU_CNTL_FRONT_CW;
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_CNTL, 1);
   OUT_RING(ring, su_cntl);
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
   OUT_RING(ring, A5XX_GRAS_SU_POINT_MINMAX_MIN(1.0f) | A5XX_GRAS_SU_POINT_MINMAX_MAX(4092.0f));
   OUT_RING(ring, A5XX_GRAS_SU_POINT_SIZE(1.0f));
   OUT_PKT4(ring, REG_A5XX_GRAS_SU_POLY_OFFSET_SCALE, 3);
   OUT_RING(ring, A5XX_GRAS_SU_POLY_OFFSET_SCALE(draw->bias_enable ? draw->bias_slope : 0));
   OUT_RING(ring, A5XX_GRAS_SU_POLY_OFFSET_OFFSET(draw->bias_enable ? draw->bias_constant : 0));
   OUT_RING(ring, A5XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP(draw->bias_enable ? draw->bias_clamp : 0));
   OUT_PKT4(ring, REG_A5XX_PC_RASTER_CNTL, 1);
   OUT_RING(ring, 0);
   OUT_PKT4(ring, REG_A5XX_GRAS_CL_CNTL, 1);
   OUT_RING(ring, A5XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z);
   /* Vulkan's provoking vertex is the first one (PC_PRIMITIVE_CNTL below). */

   uint32_t posz_regid = ir3_find_output_regid(gfx->fs.v, FRAG_RESULT_DEPTH);
   OUT_PKT4(ring, REG_A5XX_RB_FS_OUTPUT_CNTL, 1);
   OUT_RING(ring, A5XX_RB_FS_OUTPUT_CNTL_MRT(nr) |
                  COND(gfx->fs.v->writes_pos, A5XX_RB_FS_OUTPUT_CNTL_FRAG_WRITES_Z));
   OUT_PKT4(ring, REG_A5XX_SP_FS_OUTPUT_CNTL, 1);
   OUT_RING(ring, A5XX_SP_FS_OUTPUT_CNTL_MRT(nr) |
                  A5XX_SP_FS_OUTPUT_CNTL_DEPTH_REGID(posz_regid) |
                  A5XX_SP_FS_OUTPUT_CNTL_SAMPLEMASK_REGID(regid(63, 0)));

   int32_t vertex_base = draw->indexed ? draw->vertex_offset : (int32_t)draw->first_vertex;
   rd_gfx_consts(ring, device, &gfx->vs, SB4_VS_SHADER, draw, draw->descriptors[0], vertex_base);
   rd_gfx_consts(ring, device, &gfx->fs, SB4_FS_SHADER, draw, draw->descriptors[1], 0);

   /* blending */
   uint32_t mrt_blend = 0;
   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      const struct vk_color_blend_attachment_state *a =
         i < draw->cb.attachment_count ? &draw->cb.attachments[i] : NULL;
      uint32_t mask = a ? a->write_mask : 0;
      uint32_t control = A5XX_RB_MRT_CONTROL_ROP_CODE(ROP_COPY) |
                         A5XX_RB_MRT_CONTROL_COMPONENT_ENABLE(mask);
      uint32_t blend = 0;
      bool is_int = false, has_alpha = true;
      if (i < render->color_count && render->color[i].view) {
         enum pipe_format pf = vk_format_to_pipe_format(render->color[i].view->vk.format);
         is_int = util_format_is_pure_integer(pf);
         has_alpha = util_format_has_alpha(pf);
      }
      if (a && a->blend_enable && !is_int) {
         blend = A5XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR(rd_blend_factor(a->src_color_blend_factor)) |
                 A5XX_RB_MRT_BLEND_CONTROL_RGB_BLEND_OPCODE(rd_blend_op(a->color_blend_op)) |
                 A5XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR(rd_blend_factor(a->dst_color_blend_factor)) |
                 A5XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR(rd_blend_factor(a->src_alpha_blend_factor)) |
                 A5XX_RB_MRT_BLEND_CONTROL_ALPHA_BLEND_OPCODE(rd_blend_op(a->alpha_blend_op)) |
                 A5XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR(rd_blend_factor(a->dst_alpha_blend_factor));
         control |= A5XX_RB_MRT_CONTROL_BLEND | COND(has_alpha, A5XX_RB_MRT_CONTROL_BLEND2);
         mrt_blend |= 1u << i;
      }
      OUT_PKT4(ring, REG_A5XX_RB_MRT_CONTROL(i), 1);
      OUT_RING(ring, control);
      OUT_PKT4(ring, REG_A5XX_RB_MRT_BLEND_CONTROL(i), 1);
      OUT_RING(ring, blend);
   }
   OUT_PKT4(ring, REG_A5XX_SP_BLEND_CNTL, 1);
   OUT_RING(ring, A5XX_SP_BLEND_CNTL_ENABLE_BLEND(mrt_blend) | A5XX_SP_BLEND_CNTL_UNK8);
   OUT_PKT4(ring, REG_A5XX_RB_BLEND_CNTL, 1);
   OUT_RING(ring, A5XX_RB_BLEND_CNTL_ENABLE_BLEND(mrt_blend) | A5XX_RB_BLEND_CNTL_INDEPENDENT_BLEND |
                  A5XX_RB_BLEND_CNTL_SAMPLE_MASK(0xffff));
   const float *bc = draw->cb.blend_constants;
   OUT_PKT4(ring, REG_A5XX_RB_BLEND_RED, 8);
   for (unsigned c = 0; c < 4; c++) {
      OUT_RING(ring, A5XX_RB_BLEND_RED_FLOAT(bc[c]) |
                     A5XX_RB_BLEND_RED_UINT(CLAMP(bc[c], 0.f, 1.f) * 0xff) |
                     A5XX_RB_BLEND_RED_SINT(CLAMP(bc[c], -1.f, 1.f) * 0x7f));
      OUT_RING(ring, A5XX_RB_BLEND_RED_F32(bc[c]));
   }

   const struct rd_pipeline *vres = gfx->vs.res, *fres = gfx->fs.res;
   rd_emit_stage_textures(device, ring, SB4_VS_TEX, vres, draw->descriptors[0]);
   rd_emit_stage_textures(device, ring, SB4_FS_TEX, fres, draw->descriptors[1]);
   OUT_PKT4(ring, REG_A5XX_TPL1_VS_TEX_COUNT, 1);
   OUT_RING(ring, MAX2(vres->resource_usage.texture_count, vres->resource_usage.sampler_count));
   OUT_PKT4(ring, REG_A5XX_TPL1_FS_TEX_COUNT, 1);
   OUT_RING(ring, MAX2(fres->resource_usage.texture_count, fres->resource_usage.sampler_count));
   OUT_PKT4(ring, REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(ring, 0);

   rd_gfx_vertex_buffers(ring, device, gfx, draw);
   OUT_PKT4(ring, REG_A5XX_VFD_INDEX_OFFSET, 2);
   OUT_RING(ring, vertex_base);
   OUT_RING(ring, draw->first_instance);
   uint32_t index_size = draw->index_type == VK_INDEX_TYPE_UINT16 ? 2 : 4;
   bool restart = dyn->ia.primitive_restart_enable && draw->indexed;
   OUT_PKT4(ring, REG_A5XX_PC_RESTART_INDEX, 1);
   OUT_RING(ring, restart ? (index_size == 2 ? 0xffff : 0xffffffff) : 0xffffffff);
   OUT_PKT4(ring, REG_A5XX_PC_PRIMITIVE_CNTL, 1);
   OUT_RING(ring, A5XX_PC_PRIMITIVE_CNTL_STRIDE_IN_VPC(max_loc) |
                  COND(restart, A5XX_PC_PRIMITIVE_CNTL_PRIMITIVE_RESTART));
   OUT_PKT4(ring, REG_A5XX_RB_RENDER_CNTL, 1);
   OUT_RING(ring, 0x8);
   OUT_PKT4(ring, REG_A5XX_GRAS_SC_CNTL, 1);
   OUT_RING(ring, 0x8);

   uint32_t initiator = CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(rd_primtype(dyn->ia.primitive_topology)) |
                        CP_DRAW_INDX_OFFSET_0_VIS_CULL(IGNORE_VISIBILITY);
   if (draw->indexed) {
      const struct rd_buffer *ib = draw->index_buffer;
      VkDeviceSize offset = draw->index_offset + (VkDeviceSize)draw->first_index * index_size;
      uint32_t max_indices = offset < ib->vk.size ? (ib->vk.size - offset) / index_size : 0;
      OUT_PKT7(ring, CP_DRAW_INDX_OFFSET, 7);
      OUT_RING(ring, initiator | CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_DMA) |
                     CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(index_size == 2 ? INDEX4_SIZE_16_BIT :
                                                                        INDEX4_SIZE_32_BIT));
      OUT_RING(ring, draw->instance_count);
      OUT_RING(ring, draw->vertex_count);
      OUT_RING(ring, 0);
      OUT_RELOC(ring, ib->memory->bo, ib->offset + offset, 0, 0);
      OUT_RING(ring, max_indices);
   } else {
      OUT_PKT7(ring, CP_DRAW_INDX_OFFSET, 3);
      OUT_RING(ring, initiator | CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_AUTO_INDEX) |
                     CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(INDEX4_SIZE_32_BIT));
      OUT_RING(ring, draw->instance_count);
      OUT_RING(ring, draw->vertex_count);
   }
}

/* Indirect parameters are read on the CPU: commands run in order and each
 * submission completes before the next, so earlier writes are visible. */
static VkResult
rd_resolve_draw_indirect(struct rd_device *device, struct rd_draw *draw)
{
   const struct rd_buffer *buffer = draw->indirect;
   VkDeviceSize size = draw->indexed ? sizeof(VkDrawIndexedIndirectCommand) :
                                       sizeof(VkDrawIndirectCommand);
   if (!buffer || !buffer->memory || draw->indirect_offset > buffer->vk.size ||
       size > buffer->vk.size - draw->indirect_offset)
      return VK_ERROR_UNKNOWN;
   if (fd_bo_cpu_prep(buffer->memory->bo, device->pipe, FD_BO_PREP_READ))
      return VK_ERROR_DEVICE_LOST;
   const uint8_t *map = fd_bo_map(buffer->memory->bo);
   if (!map)
      return VK_ERROR_MEMORY_MAP_FAILED;
   map += buffer->offset + draw->indirect_offset;
   if (draw->indexed) {
      VkDrawIndexedIndirectCommand c;
      memcpy(&c, map, sizeof(c));
      draw->vertex_count = c.indexCount;
      draw->instance_count = c.instanceCount;
      draw->first_index = c.firstIndex;
      draw->vertex_offset = c.vertexOffset;
      draw->first_instance = c.firstInstance;
   } else {
      VkDrawIndirectCommand c;
      memcpy(&c, map, sizeof(c));
      draw->vertex_count = c.vertexCount;
      draw->instance_count = c.instanceCount;
      draw->first_vertex = c.firstVertex;
      draw->first_instance = c.firstInstance;
   }
   draw->indirect = NULL;
   return VK_SUCCESS;
}

VkResult
rd_execute_render(struct rd_device *device, const struct rd_render *render)
{
   if (render->view_mask)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   for (const struct rd_render_attachment *a = &render->depth; a <= &render->stencil; a++) {
      if (!a->view)
         continue;
      if (a->load == VK_ATTACHMENT_LOAD_OP_CLEAR)
         return VK_ERROR_FEATURE_NOT_PRESENT; /* load-op clears come with vk_meta */
      const struct rd_image *image = container_of(a->view->vk.image, struct rd_image, vk);
      if (!image->memory || image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   uint32_t width = 0, height = 0;
   for (unsigned i = 0; i < render->color_count; i++) {
      const struct rd_render_attachment *a = &render->color[i];
      if (!a->view)
         continue;
      if (a->load == VK_ATTACHMENT_LOAD_OP_CLEAR)
         return VK_ERROR_FEATURE_NOT_PRESENT; /* load-op clears come with vk_meta */
      const struct rd_image *image = container_of(a->view->vk.image, struct rd_image, vk);
      if (!image->memory || image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      width = MAX2(width, a->view->vk.extent.width);
      height = MAX2(height, a->view->vk.extent.height);
   }
   for (const struct rd_render_attachment *a = &render->depth; a <= &render->stencil; a++) {
      if (a->view) {
         width = MAX2(width, a->view->vk.extent.width);
         height = MAX2(height, a->view->vk.extent.height);
      }
   }
   if (!width || !util_dynarray_num_elements(&render->draws, struct rd_draw))
      return VK_SUCCESS;
   struct fd_submit *submit = fd_submit_new(device->pipe);
   if (!submit)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct fd_ringbuffer *ring = fd_submit_new_ringbuffer(submit, 0,
      FD_RINGBUFFER_PRIMARY | FD_RINGBUFFER_GROWABLE);
   if (!ring) {
      fd_submit_del(submit);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   rd_gfx_restore(ring, device);
   rd_gfx_sysmem_prep(ring, device, render, width, height);
   VkResult result = VK_SUCCESS;
   util_dynarray_foreach(&render->draws, struct rd_draw, recorded) {
      struct rd_draw draw = *recorded;
      if (draw.indirect && (result = rd_resolve_draw_indirect(device, &draw)) != VK_SUCCESS)
         break;
      if (!draw.vertex_count || !draw.instance_count)
         continue;
      if (draw.indexed && (!draw.index_buffer || !draw.index_buffer->memory)) {
         result = VK_ERROR_UNKNOWN;
         break;
      }
      fd_ringbuffer_attach_bo(ring, draw.pipeline->gfx->vs.bo);
      fd_ringbuffer_attach_bo(ring, draw.pipeline->gfx->fs.bo);
      rd_gfx_draw(ring, device, render, &draw);
   }
   if (result != VK_SUCCESS) {
      fd_ringbuffer_del(ring);
      fd_submit_del(submit);
      return result;
   }
   OUT_PKT7(ring, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
   OUT_RING(ring, 0);
   rd_event(ring, device, PC_CCU_FLUSH_COLOR_TS, true);
   rd_event(ring, device, PC_CCU_FLUSH_DEPTH_TS, true);
   rd_flush(device, ring);
   return rd_finish_submit(device, submit, ring, NULL, NULL);
}
