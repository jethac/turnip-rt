#ifndef RD_PRIVATE_H
#define RD_PRIVATE_H

#include "rd_entrypoints.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "rd_image.h"
#include "rd_nir_resources.h"
#include "rd_sampling.h"
#include "vk_device.h"
#include "vk_device_memory.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_query_pool.h"
#include "util/u_thread.h"
#include "util/simple_mtx.h"
#include "vk_descriptor_set_layout.h"
#include "vk_pipeline_layout.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_graphics_state.h"
#include "ir3/ir3_shader.h"
#include "drm/freedreno_drmif.h"
#include "common/freedreno_dev_info.h"
#include "common/fd5_hw.h"
#include "util/u_debug.h"
#include <stdio.h>

struct fd_ringbuffer;
struct fd_submit;

struct rd_instance { struct vk_instance vk; };
struct rd_physical_device {
   struct vk_physical_device vk;
   struct fd_device *fd;
   int render_fd;
   struct vk_sync_type sync_type;
   VkResult (*sync_wait_many)(struct vk_device *device, uint32_t count,
                              const struct vk_sync_wait *waits,
                              enum vk_sync_wait_flags flags, uint64_t deadline);
   const struct vk_sync_type *sync_types[2];
};
struct rd_device {
   struct vk_device vk;
   struct fd_device *fd;
   struct fd_pipe *pipe;
   struct vk_queue queue;
   struct fd_dev_info info;
   struct ir3_compiler *compiler;
   struct fd_bo *control;
   struct fd_bo *border_colors;
   uint32_t border_colors_offset;
   uint32_t seqno;
   uint64_t submit_count;
   uint64_t faults;
   struct list_head active_queries;
   uint32_t queue_priority;
   struct rd_trace *trace;
   simple_mtx_t meta_mutex;
   struct rd_pipeline *clear_pipeline;
   struct rd_pipeline *resolve_pipeline[4];
   struct rd_pipeline *copy_texel_pipeline;
   struct rd_pipeline *blit_pipeline[9]; /* [dim 1D/2D/3D][float/int/uint] */
};
struct rd_memory {
   struct vk_device_memory vk;
   struct fd_bo *bo;
   void *map;
};
struct rd_buffer {
   struct vk_buffer vk;
   struct rd_memory *memory;
   VkDeviceSize offset;
};

VK_DEFINE_HANDLE_CASTS(rd_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(rd_physical_device, vk.base, VkPhysicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(rd_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)

#define RD_MAX_BINDINGS 1024
#define RD_MAX_ACTIVE_RESOURCES 64
#define RD_MAX_SETS 4
#define RD_MAX_SHARED_SIZE 16384
/* Workgroup invocation limit (barrier shaders must keep the whole group
 * resident: ir3 RA bounds their register footprint accordingly). */
#define RD_MAX_WORKGROUP_INVOCATIONS 256
struct rd_binding { uint32_t binding, count, slot; VkDescriptorType type; VkShaderStageFlags stages; uint32_t immutable_offset; };
struct rd_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count, descriptor_count;
   uint32_t immutable_count;
   struct vk_sampler_state *immutable_sampler_states;
   struct rd_binding bindings[];
};
struct rd_descriptor {
   VkDescriptorType type;
   struct rd_buffer *buffer;
   VkDeviceSize offset, range;
   struct rd_image_view *image_view;
   struct rd_buffer_view *buffer_view;
   bool has_sampler;
   uint32_t sampler[4];
};
struct rd_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;
   VkAllocationCallbacks alloc;
   uint32_t max_sets, used_sets, max_descriptors, used_descriptors;
};
struct rd_descriptor_set {
   struct vk_object_base base;
   struct list_head link;
   struct rd_descriptor_pool *pool;
   struct rd_set_layout *layout;
   struct rd_descriptor descriptors[];
};
struct rd_pipeline;
struct rd_gfx_shader {
   struct ir3_shader_variant *v;
   struct fd_bo *bo;
   /* per-stage resource slots (only the resource fields are used) */
   struct rd_pipeline *res;
};
/* Graphics part of a pipeline (rd_graphics.c). */
struct rd_gfx_pipeline {
   struct rd_gfx_shader vs, fs;
   struct vk_graphics_pipeline_all_state all;
   struct vk_graphics_pipeline_state state;
   struct vk_dynamic_graphics_state dynamic;
};
struct rd_pipeline {
   struct vk_object_base base;
   struct rd_gfx_pipeline *gfx;
   struct vk_pipeline_layout *layout;
   struct ir3_shader_variant *variant;
   struct fd_bo *bo, *private_bo;
   uint32_t private_per_sp;
   uint32_t descriptor_count;
   uint64_t used_descriptors;
   uint32_t logical_local_size[3];
   bool linear_subgroups;
   bool packed_1d_barrier;
   bool linear_barriers;
   bool batch_local_memory;
   VkPipelineCreateFlags compile_flags;
   struct nir_shader *sampling_nir;
   struct rd_sampling_entry *sampling_variants;
   uint32_t sampling_mask, sampling_buffer;
   struct rd_nir_resource_binding resources[RD_MAX_ACTIVE_RESOURCES];
   uint32_t resource_count;
   struct rd_nir_resource_usage resource_usage;
   uint32_t texture_descriptors[16], sampler_descriptors[16], image_descriptors[16];
   uint32_t buffer_count, buffer_descriptors[32];
};
struct rd_dispatch {
   struct rd_pipeline *pipeline;
   struct rd_buffer *indirect;
   struct fd_bo *sampling_addresses, *sampling_bos[16];
   VkDeviceSize indirect_offset;
   struct rd_descriptor descriptors[RD_MAX_ACTIVE_RESOURCES];
   struct rd_descriptor_set *sets[RD_MAX_SETS];
   uint32_t dynamic_offsets[RD_MAX_ACTIVE_RESOURCES];
   uint32_t push[32], grid[3], base[3];
};
struct rd_event {
   struct vk_object_base base;
   mtx_t mutex;
   cnd_t condition;
   bool signaled;
};
struct rd_query_pool;
struct rd_query_slot {
   struct list_head active_link;
   struct rd_query_pool *pool;
   uint64_t values[11];
   bool available;
   bool active;
};
struct rd_query_pool {
   struct vk_query_pool vk;
   mtx_t mutex;
   cnd_t condition;
   struct fd_bo *bo;
   void *map;
   uint32_t value_count;
   struct rd_query_slot slots[];
};
struct rd_event_command { struct rd_event *event; };
#define RD_MAX_RTS 8
struct rd_render_attachment {
   struct rd_image_view *view;
   struct rd_image_view *resolve;
   VkResolveModeFlagBits resolve_mode;
   VkAttachmentLoadOp load;
   VkAttachmentStoreOp store;
   VkClearValue clear;
};
struct rd_draw {
   struct rd_pipeline *pipeline;
   uint32_t vertex_count, instance_count, first_vertex, first_instance;
   /* indexed draws: vertex_count is the index count */
   bool indexed;
   uint32_t first_index;
   int32_t vertex_offset;
   struct rd_buffer *index_buffer;
   VkDeviceSize index_offset;
   VkIndexType index_type;
   /* indirect draws: the parameters are read from this buffer at submit */
   struct rd_buffer *indirect;
   VkDeviceSize indirect_offset;
   VkViewport viewport;
   VkRect2D scissor;
   VkCullModeFlags cull;
   VkFrontFace front;
   struct vk_depth_stencil_state ds;
   struct vk_color_blend_state cb;
   bool bias_enable;
   float bias_constant, bias_clamp, bias_slope, line_width;
   uint16_t sample_mask;
   bool alpha_to_coverage;
   struct rd_buffer *vertex_buffers[32];
   VkDeviceSize vertex_offsets[32];
   uint32_t push[32];
   /* descriptors of each stage's resources, copied when recorded */
   struct rd_descriptor *descriptors[2];
   /* the occlusion query active when the draw was recorded */
   struct rd_query_pool *occlusion_pool;
   uint32_t occlusion_query;
   /* a pipeline barrier inside the render pass precedes this draw */
   bool barrier;
};
/* vkCmdBeginQuery/vkCmdEndQuery of an occlusion query inside a render */
struct rd_render_query {
   struct rd_query_pool *pool;
   uint32_t query;
   bool end;
};
/* One dynamic-rendering instance: its attachments and its draws. */
struct rd_render {
   VkRect2D area;
   uint32_t layers, view_mask, color_count, samples;
   uint32_t view_index; /* during a multiview replay */
   struct rd_render_attachment color[RD_MAX_RTS], depth, stencil;
   /* draws of a secondary command buffer recorded inside a render pass */
   bool continued;
   struct util_dynarray draws;
   struct util_dynarray queries; /* struct rd_render_query */
};
struct rd_query_command {
   struct rd_query_pool *pool;
   struct rd_buffer *dst;
   uint32_t first, count;
   VkDeviceSize offset, stride;
   VkQueryResultFlags flags;
   VkPipelineStageFlags2 stage;
};
enum rd_command_type {
   RD_CMD_DISPATCH, RD_CMD_COPY, RD_CMD_FILL, RD_CMD_UPDATE,
   RD_CMD_IMAGE_COPY, RD_CMD_IMAGE_CLEAR,
   RD_CMD_EVENT_SET, RD_CMD_EVENT_RESET, RD_CMD_EVENT_WAIT,
   RD_CMD_QUERY_RESET, RD_CMD_QUERY_BEGIN, RD_CMD_QUERY_END,
   RD_CMD_QUERY_TIMESTAMP, RD_CMD_QUERY_COPY,
   RD_CMD_RENDER, RD_CMD_BLIT,
};
/* Image operations done as internal renders: one region of vkCmdBlitImage,
 * vkCmdResolveImage or one range of vkCmdClearDepthStencilImage. */
enum rd_blit_kind { RD_BLIT_SCALED, RD_BLIT_RESOLVE, RD_BLIT_CLEAR_DS };
struct rd_blit {
   enum rd_blit_kind kind;
   struct rd_image *src, *dst;
   uint32_t src_level, dst_level, src_layer, dst_layer, layers, levels;
   int32_t src_offsets[2][3], dst_offsets[2][3];
   VkFilter filter;
   VkImageAspectFlags aspects;
   VkClearDepthStencilValue clear;
};
struct rd_transfer {
   struct rd_buffer *src, *dst;
   VkDeviceSize src_offset, dst_offset, size;
   uint32_t value;
   void *data;
};
struct rd_command {
   enum rd_command_type type;
   union {
      struct rd_dispatch dispatch;
      struct rd_transfer transfer;
      struct rd_image_transfer image_transfer;
      struct rd_event_command event;
      struct rd_query_command query;
      struct rd_render *render;
      struct rd_blit blit;
   };
};
struct rd_command_buffer {
   struct vk_command_buffer vk;
   struct rd_pipeline *pipeline;
   struct rd_pipeline *gfx_pipeline;
   struct rd_render *render;
   bool render_continue;
   struct rd_query_pool *occlusion_pool;
   uint32_t occlusion_query;
   bool render_barrier;
   /* bound descriptor sets: [0] compute, [1] graphics */
   struct rd_bind_state {
      struct rd_descriptor_set *sets[RD_MAX_SETS];
      uint32_t dynamic_offsets[RD_MAX_SETS][RD_MAX_BINDINGS];
      uint8_t set_compatibility[RD_MAX_SETS][RD_MAX_SETS][BLAKE3_OUT_LEN];
   } bind[2];
   struct rd_buffer *vertex_buffers[32];
   VkDeviceSize vertex_offsets[32];
   struct rd_buffer *index_buffer;
   VkDeviceSize index_offset;
   VkIndexType index_type;
   uint32_t push[32];
   struct util_dynarray commands;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_set_layout, vk.base, VkDescriptorSetLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_descriptor_pool, base, VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_descriptor_set, base, VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_pipeline, base, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_HANDLE_CASTS(rd_command_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)
extern const struct vk_command_buffer_ops rd_command_buffer_ops;
struct rd_command *rd_command_append(struct rd_command_buffer *command,
                                     enum rd_command_type type);
bool rd_shader_format_supported(VkFormat format);
bool rd_storage_shader_format_supported(VkFormat format);
bool rd_shader_format_filterable(VkFormat format);
VkResult rd_init_border_colors(struct rd_device *device);
/* rd_queue.c, shared with rd_graphics.c */
void rd_flush(struct rd_device *device, struct fd_ringbuffer *ring);
struct rd_gpu_elapsed;
struct rd_submit_profile;
VkResult rd_finish_submit(struct rd_device *device, struct fd_submit *submit,
                          struct fd_ringbuffer *ring, struct rd_submit_profile *profile,
                          struct rd_gpu_elapsed *sample);
void rd_emit_stage_textures(struct rd_device *device, struct fd_ringbuffer *ring,
                            enum a4xx_state_block sb, const struct rd_pipeline *stage,
                            const struct rd_descriptor *descriptors);
const struct rd_nir_resource_binding *rd_pipeline_resource(void *data, uint32_t set,
                                                           uint32_t binding, uint32_t element);
bool rd_lower_stage_buffers(struct nir_shader *nir, struct rd_pipeline *stage);
/* rd_graphics.c */
bool rd_graphics_enabled(void);
void rd_meta_finish(struct rd_device *device);
VkResult rd_execute_render(struct rd_device *device, const struct rd_render *render);
VkResult rd_execute_blit(struct rd_device *device, const struct rd_blit *blit);
void rd_render_free(struct rd_command_buffer *command, struct rd_render *render);
bool rd_render_query(struct rd_command_buffer *command, struct rd_query_pool *pool,
                     uint32_t query, bool end);
bool rd_render_execute(struct rd_command_buffer *primary, const struct rd_render *render);
void rd_gfx_pipeline_destroy(struct rd_device *device, struct rd_pipeline *pipeline);
void rd_sampler_descriptor(const struct vk_sampler_state *state, uint32_t words[4]);
bool rd_descriptor_ready(const struct rd_descriptor *descriptor);
void rd_emit_shader_resources(struct rd_device *device, struct fd_ringbuffer *ring,
                              const struct rd_dispatch *dispatch);
void rd_emit_image_constants(struct fd_ringbuffer *ring,
                             const struct rd_dispatch *dispatch);
VkResult rd_queue_submit(struct vk_queue *queue, struct vk_queue_submit *submit);
VkResult rd_compile_pipeline(struct rd_device *device, struct rd_pipeline *pipeline,
                             const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags);
void rd_wake_device_waiters(struct vk_device *device);
VkResult rd_sync_wait_many(struct vk_device *device, uint32_t count,
                           const struct vk_sync_wait *waits,
                           enum vk_sync_wait_flags flags, uint64_t deadline);
VkResult rd_execute_event(struct rd_device *device, enum rd_command_type type,
                          const struct rd_event_command *command);
VkResult rd_execute_query(struct rd_device *device, enum rd_command_type type,
                          const struct rd_query_command *command);
VkResult rd_query_timestamp(struct rd_device *device, struct rd_query_pool *pool,
                            uint64_t *value);
void rd_query_account_dispatch(struct rd_device *device,
                                const struct rd_dispatch *dispatch);
#ifdef __cplusplus
extern "C" {
#endif
bool rd_pipeline_cache_restore(struct rd_device *device, struct rd_pipeline *pipeline,
   VkPipelineCache cache, const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags);
void rd_pipeline_cache_store(struct rd_device *device, struct rd_pipeline *pipeline,
   VkPipelineCache cache, const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags);
/* vk_command_buffer_set_error with the recording site printed when
 * RADISH_DEBUG_ERRORS is set (which command made vkEndCommandBuffer fail). */
#define rd_set_error(command_vk, result)                                        \
   do {                                                                         \
      if (debug_get_bool_option("RADISH_DEBUG_ERRORS", false))                 \
         fprintf(stderr, "radish: command error %d at %s:%d (%s)\n", (result),  \
                 __FILE__, __LINE__, __func__);                                 \
      vk_command_buffer_set_error((command_vk), (result));                      \
   } while (0)
#ifdef __cplusplus
}
#endif
#endif
