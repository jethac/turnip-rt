#ifndef RD_IMAGE_H
#define RD_IMAGE_H

#include "vk_image.h"
#include "fdl/freedreno_layout.h"
#include "vk_buffer_view.h"
#include "vk_sampler.h"

#define RD_IMAGE_MAX_LEVELS 13
#define RD_IMAGE_MAX_SIZE (UINT32_MAX - 4095ull)

struct rd_device;
struct rd_memory;
struct rd_buffer;
struct rd_command_buffer;

struct rd_image_level {
   VkDeviceSize offset, row_pitch, depth_pitch, array_pitch, size;
};

struct rd_image {
   struct vk_image vk;
   struct rd_memory *memory;
   VkDeviceSize memory_offset, size;
   uint32_t texel_size;
   struct fdl_layout native_layout;
   struct rd_image_level levels[RD_IMAGE_MAX_LEVELS];
};

struct rd_image_view { struct vk_image_view vk; };
struct rd_buffer_view { struct vk_buffer_view vk; };
struct rd_sampler {
   struct vk_sampler vk;
   struct vk_sampler_state state;
};

struct rd_image_transfer {
   struct rd_image *src_image, *dst_image;
   struct rd_buffer *src_buffer, *dst_buffer;
   VkDeviceSize src_offset, dst_offset;
   VkDeviceSize src_row_pitch, dst_row_pitch;
   VkDeviceSize src_slice_pitch, dst_slice_pitch;
   VkDeviceSize row_bytes;
   uint32_t rows, slices, texel_size, copy_mask;
   uint8_t clear[16];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(rd_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_image_view, vk.base, VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_buffer_view, vk.base, VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW)
VK_DEFINE_NONDISP_HANDLE_CASTS(rd_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER)

VkResult rd_execute_image_transfer(struct rd_device *device,
                                   const struct rd_image_transfer *transfer,
                                   bool clear);
bool rd_image_barrier_valid(const VkImageMemoryBarrier2 *barrier);
#endif
