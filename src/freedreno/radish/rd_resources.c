#include "rd_private.h"
#include "common/fd5_hw.h"
#include "common/freedreno_common.h"
#include "drm/freedreno_ringbuffer.h"
#include "vk_format.h"
#include "util/u_math.h"
#include "util/u_debug.h"

struct rd_resource {
   struct fd_bo *bo;
   VkDeviceSize offset;
   enum a5xx_tex_fmt format;
   enum a5xx_tex_type type;
   uint32_t cpp, width, height, depth, row_pitch, array_pitch, format_word;
   uint32_t mip_levels, pitchalign, min_layer_size;
   bool buffer, image_size_only;
};

struct rd_border_color {
   uint32_t fp32[4];
   uint16_t ui16[4];
   int16_t si16[4];
   uint16_t fp16[4];
   uint16_t rgb565, rgb5a1, rgba4;
   uint8_t pad0[2], ui8[4];
   int8_t si8[4];
   uint32_t rgb10a2, z24;
   uint16_t srgb[4];
   uint8_t pad1[56];
};

_Static_assert(sizeof(struct rd_border_color) == 128, "a5xx border color entry");

VkResult
rd_init_border_colors(struct rd_device *device)
{
   device->border_colors = fd_bo_new(device->fd, 16384, FD_BO_GPUREADONLY,
                                      "radish-border-colors");
   if (!device->border_colors)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   void *map = fd_bo_map(device->border_colors);
   if (!map) {
      fd_bo_del(device->border_colors);
      device->border_colors = NULL;
      return VK_ERROR_MEMORY_MAP_FAILED;
   }
   uint64_t address = fd_bo_get_iova(device->border_colors);
   device->border_colors_offset = align64(address, 8192) - address;
   struct rd_border_color *colors = (void *)((uint8_t *)map + device->border_colors_offset);
   memset(colors, 0, 6 * sizeof(*colors));
   for (uint32_t i = 0; i < 6; i++) {
      bool integer = i & 1;
      struct rd_border_color *color = &colors[i];
      for (uint32_t j = 0; j < 4; j++) {
         uint32_t value = i >= 4 || (i >= 2 && j == 3);
         color->fp32[j] = integer ? value : value ? 0x3f800000 : 0;
         color->fp16[j] = integer ? value : value ? 0x3c00 : 0;
         if (integer)
            continue;
         color->ui16[j] = value ? 0xffff : 0;
         color->si16[j] = value ? 0x7fff : 0;
         color->ui8[j] = value ? 0xff : 0;
         color->si8[j] = value ? 0x7f : 0;
         color->srgb[j] = value ? 0x3c00 : 0;
         if (j == 1)
            color->rgb565 |= value ? 0x7e0 : 0;
         else if (j < 3)
            color->rgb565 |= value ? 0x1f << (j ? 11 : 0) : 0;
         if (j == 3) {
            color->rgb5a1 |= value ? 0x8000 : 0;
            color->rgb10a2 |= value ? 3u << 30 : 0;
         } else {
            color->rgb5a1 |= value ? 0x1f << (j * 5) : 0;
            color->rgb10a2 |= value ? 0x3ffu << (j * 10) : 0;
         }
         color->rgba4 |= value ? 0xf << (j * 4) : 0;
         if (j == 0)
            color->z24 = value ? 0xffffff : 0;
      }
   }
   __sync_synchronize();
   return VK_SUCCESS;
}

static enum a5xx_tex_fmt
rd_resource_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM: return TFMT5_8_UNORM;
   case VK_FORMAT_R8_SNORM: return TFMT5_8_SNORM;
   case VK_FORMAT_R8_UINT: return TFMT5_8_UINT;
   case VK_FORMAT_R8_SINT: return TFMT5_8_SINT;
   case VK_FORMAT_R8G8_UNORM: return TFMT5_8_8_UNORM;
   case VK_FORMAT_R8G8_SNORM: return TFMT5_8_8_SNORM;
   case VK_FORMAT_R8G8_UINT: return TFMT5_8_8_UINT;
   case VK_FORMAT_R8G8_SINT: return TFMT5_8_8_SINT;
   case VK_FORMAT_R8G8B8A8_UNORM: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_R8G8B8A8_SNORM: return TFMT5_8_8_8_8_SNORM;
   case VK_FORMAT_R8G8B8A8_UINT: return TFMT5_8_8_8_8_UINT;
   case VK_FORMAT_R8G8B8A8_SINT: return TFMT5_8_8_8_8_SINT;
   case VK_FORMAT_R8G8B8A8_SRGB: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_B8G8R8A8_UNORM: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_B8G8R8A8_SRGB: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_R16_UNORM: return TFMT5_16_UNORM;
   case VK_FORMAT_R16_SNORM: return TFMT5_16_SNORM;
   case VK_FORMAT_R16_UINT: return TFMT5_16_UINT;
   case VK_FORMAT_R16_SINT: return TFMT5_16_SINT;
   case VK_FORMAT_R16_SFLOAT: return TFMT5_16_FLOAT;
   case VK_FORMAT_R16G16_UNORM: return TFMT5_16_16_UNORM;
   case VK_FORMAT_R16G16_SNORM: return TFMT5_16_16_SNORM;
   case VK_FORMAT_R16G16_UINT: return TFMT5_16_16_UINT;
   case VK_FORMAT_R16G16_SINT: return TFMT5_16_16_SINT;
   case VK_FORMAT_R16G16_SFLOAT: return TFMT5_16_16_FLOAT;
   case VK_FORMAT_R16G16B16A16_UNORM: return TFMT5_16_16_16_16_UNORM;
   case VK_FORMAT_R16G16B16A16_SNORM: return TFMT5_16_16_16_16_SNORM;
   case VK_FORMAT_R16G16B16A16_UINT: return TFMT5_16_16_16_16_UINT;
   case VK_FORMAT_R16G16B16A16_SINT: return TFMT5_16_16_16_16_SINT;
   case VK_FORMAT_R16G16B16A16_SFLOAT: return TFMT5_16_16_16_16_FLOAT;
   case VK_FORMAT_R32_UINT: return TFMT5_32_UINT;
   case VK_FORMAT_R32_SINT: return TFMT5_32_SINT;
   case VK_FORMAT_R32_SFLOAT: return TFMT5_32_FLOAT;
   case VK_FORMAT_R32G32_UINT: return TFMT5_32_32_UINT;
   case VK_FORMAT_R32G32_SINT: return TFMT5_32_32_SINT;
   case VK_FORMAT_R32G32_SFLOAT: return TFMT5_32_32_FLOAT;
   case VK_FORMAT_R32G32B32A32_UINT: return TFMT5_32_32_32_32_UINT;
   case VK_FORMAT_R32G32B32A32_SINT: return TFMT5_32_32_32_32_SINT;
   case VK_FORMAT_R32G32B32A32_SFLOAT: return TFMT5_32_32_32_32_FLOAT;
   case VK_FORMAT_R5G6B5_UNORM_PACK16: return TFMT5_5_6_5_UNORM;
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16: return TFMT5_5_5_5_1_UNORM;
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return TFMT5_8_8_8_8_SNORM;
   case VK_FORMAT_A8B8G8R8_UINT_PACK32: return TFMT5_8_8_8_8_UINT;
   case VK_FORMAT_A8B8G8R8_SINT_PACK32: return TFMT5_8_8_8_8_SINT;
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return TFMT5_8_8_8_8_UNORM;
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return TFMT5_10_10_10_2_UNORM;
   case VK_FORMAT_A2R10G10B10_UINT_PACK32: return TFMT5_10_10_10_2_UINT;
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return TFMT5_10_10_10_2_UNORM;
   case VK_FORMAT_A2B10G10R10_UINT_PACK32: return TFMT5_10_10_10_2_UINT;
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return TFMT5_11_11_10_FLOAT;
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return TFMT5_9_9_9_E5_FLOAT;
   case VK_FORMAT_R32G32B32_UINT: return TFMT5_32_32_32_UINT;
   case VK_FORMAT_R32G32B32_SINT: return TFMT5_32_32_32_SINT;
   case VK_FORMAT_R32G32B32_SFLOAT: return TFMT5_32_32_32_FLOAT;
   case VK_FORMAT_R4G4B4A4_UNORM_PACK16: return TFMT5_4_4_4_4_UNORM;
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return TFMT5_4_4_4_4_UNORM;
   case VK_FORMAT_B5G6R5_UNORM_PACK16: return TFMT5_5_6_5_UNORM;
   case VK_FORMAT_B8G8R8A8_SNORM: return TFMT5_8_8_8_8_SNORM;
   case VK_FORMAT_B8G8R8A8_UINT: return TFMT5_8_8_8_8_UINT;
   case VK_FORMAT_B8G8R8A8_SINT: return TFMT5_8_8_8_8_SINT;
   case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: return TFMT5_ETC2_RGB8;
   case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK: return TFMT5_ETC2_RGB8;
   case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK: return TFMT5_ETC2_RGB8A1;
   case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK: return TFMT5_ETC2_RGB8A1;
   case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: return TFMT5_ETC2_RGBA8;
   case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK: return TFMT5_ETC2_RGBA8;
   case VK_FORMAT_EAC_R11_UNORM_BLOCK: return TFMT5_ETC2_R11_UNORM;
   case VK_FORMAT_EAC_R11_SNORM_BLOCK: return TFMT5_ETC2_R11_SNORM;
   case VK_FORMAT_EAC_R11G11_UNORM_BLOCK: return TFMT5_ETC2_RG11_UNORM;
   case VK_FORMAT_EAC_R11G11_SNORM_BLOCK: return TFMT5_ETC2_RG11_SNORM;
   case VK_FORMAT_D16_UNORM: return TFMT5_16_UNORM;
   case VK_FORMAT_D32_SFLOAT: return TFMT5_32_FLOAT;
   case VK_FORMAT_S8_UINT: return TFMT5_8_UINT;
   case VK_FORMAT_D24_UNORM_S8_UINT: return TFMT5_X8Z24_UNORM;
   default: return TFMT5_NONE;
   }
}

bool
rd_shader_format_supported(VkFormat format)
{
   return rd_resource_format(format) != TFMT5_NONE;
}

bool
rd_storage_shader_format_supported(VkFormat format)
{
   if (vk_format_is_depth_or_stencil(format) || vk_format_is_compressed(format))
      return false;
   switch (format) {
   case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   case VK_FORMAT_B8G8R8A8_SNORM:
   case VK_FORMAT_B8G8R8A8_UINT:
   case VK_FORMAT_B8G8R8A8_SINT:

   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   case VK_FORMAT_A2R10G10B10_UINT_PACK32:
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
   case VK_FORMAT_R32G32B32_UINT:
   case VK_FORMAT_R32G32B32_SINT:
   case VK_FORMAT_R32G32B32_SFLOAT:
      return false;
   default:
      return rd_shader_format_supported(format) && !vk_format_is_srgb(format) &&
             format != VK_FORMAT_B8G8R8A8_UNORM;
   }
}

bool
rd_shader_format_filterable(VkFormat format)
{
   return rd_shader_format_supported(format) && !vk_format_is_int(format) &&
          !vk_format_has_stencil(format);
}

static enum a5xx_tex_swiz
rd_resource_swizzle(unsigned swizzle)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X: return A5XX_TEX_X;
   case PIPE_SWIZZLE_Y: return A5XX_TEX_Y;
   case PIPE_SWIZZLE_Z: return A5XX_TEX_Z;
   case PIPE_SWIZZLE_W: return A5XX_TEX_W;
   case PIPE_SWIZZLE_1: return A5XX_TEX_ONE;
   default: return A5XX_TEX_ZERO;
   }
}

static unsigned char
rd_view_swizzle(VkComponentSwizzle swizzle)
{
   switch (swizzle) {
   case VK_COMPONENT_SWIZZLE_R: return PIPE_SWIZZLE_X;
   case VK_COMPONENT_SWIZZLE_G: return PIPE_SWIZZLE_Y;
   case VK_COMPONENT_SWIZZLE_B: return PIPE_SWIZZLE_Z;
   case VK_COMPONENT_SWIZZLE_A: return PIPE_SWIZZLE_W;
   case VK_COMPONENT_SWIZZLE_ONE: return PIPE_SWIZZLE_1;
   default: return PIPE_SWIZZLE_0;
   }
}

static bool
rd_make_resource(const struct rd_descriptor *descriptor, struct rd_resource *out)
{
   memset(out, 0, sizeof(*out));
   VkFormat format;
   enum a3xx_msaa_samples samples = MSAA_ONE;
   if (descriptor->image_view) {
      const struct vk_image_view *view = &descriptor->image_view->vk;
      const struct rd_image *image = container_of(view->image, struct rd_image, vk);
      if (!image->memory)
         return false;
      const struct rd_image_level *level = &image->levels[view->base_mip_level];
      VkExtent3D extent = view->extent;
      out->bo = image->memory->bo;
      out->offset = image->memory_offset + level->offset +
                    view->base_array_layer * level->array_pitch;
      format = view->format;
      samples = image->vk.samples == VK_SAMPLE_COUNT_4_BIT ? MSAA_FOUR : MSAA_ONE;
      out->format = rd_resource_format(format);
      if (format == VK_FORMAT_D24_UNORM_S8_UINT && view->aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
         out->format = TFMT5_8_8_8_8_UINT;
      out->cpp = image->texel_size;
      out->width = extent.width;
      out->height = extent.height;
      switch (view->view_type) {
      case VK_IMAGE_VIEW_TYPE_1D:
      case VK_IMAGE_VIEW_TYPE_1D_ARRAY: out->type = A5XX_TEX_1D; break;
      case VK_IMAGE_VIEW_TYPE_3D: out->type = A5XX_TEX_3D; break;
      case VK_IMAGE_VIEW_TYPE_CUBE:
      case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: out->type = A5XX_TEX_CUBE; break;
      default: out->type = A5XX_TEX_2D; break;
      }
      out->depth = view->view_type == VK_IMAGE_VIEW_TYPE_3D ? extent.depth :
                   out->type == A5XX_TEX_CUBE ? view->layer_count / 6 : view->layer_count;
      if (descriptor->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && out->type == A5XX_TEX_CUBE) {
         out->type = A5XX_TEX_2D;
         out->depth = view->layer_count;
      }
      /* A 1D-array storage image is addressed as (x, layer), with the layer
       * in the second coordinate (the byte-offset constants already use the
       * array pitch as the y pitch), so the layer count is its height: the
       * ldib/stib bounds check against HEIGHT=1 dropped every layer but 0.
       */
      if (descriptor->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
          view->view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY)
         out->height = view->layer_count;
      out->mip_levels = view->level_count - 1;
      out->image_size_only = descriptor->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      out->pitchalign = image->native_layout.pitchalign - 6;
      out->min_layer_size = image->native_layout.slices[image->vk.mip_levels - 1].size0;
      out->row_pitch = level->row_pitch;
      out->array_pitch = view->view_type == VK_IMAGE_VIEW_TYPE_3D ?
                         level->depth_pitch : level->array_pitch;
   } else {
      const struct rd_buffer *buffer;
      VkDeviceSize offset, elements;
      if (descriptor->buffer_view) {
         const struct vk_buffer_view *view = &descriptor->buffer_view->vk;
         buffer = container_of(view->buffer, struct rd_buffer, vk);
         format = view->format;
         offset = view->offset;
         elements = view->elements;
      } else if (descriptor->buffer) {
         buffer = descriptor->buffer;
         format = VK_FORMAT_R32_UINT;
         offset = descriptor->offset;
         elements = descriptor->range / 4;
      } else {
         return false;
      }
      if (!buffer->memory)
         return false;
      out->bo = buffer->memory->bo;
      out->offset = buffer->offset + offset;
      out->format = rd_resource_format(format);
      out->cpp = vk_format_get_blocksize(format);
      out->width = elements & BITFIELD_MASK(15);
      out->height = elements >> 15;
      out->type = A5XX_TEX_BUFFER;
      out->buffer = true;
   }
   VkFormat sampled_format = format == VK_FORMAT_D16_UNORM ? VK_FORMAT_R16_UNORM :
                             format == VK_FORMAT_D32_SFLOAT ? VK_FORMAT_R32_SFLOAT :
                             format == VK_FORMAT_D24_UNORM_S8_UINT ? VK_FORMAT_R32_SFLOAT :
                             format == VK_FORMAT_S8_UINT ? VK_FORMAT_R8_UINT : format;
   const struct util_format_description *description = vk_format_description(sampled_format);
   unsigned char swizzle[4];
   memcpy(swizzle, description->swizzle, sizeof(swizzle));
   if (format == VK_FORMAT_D24_UNORM_S8_UINT && descriptor->image_view &&
       descriptor->image_view->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      swizzle[0] = PIPE_SWIZZLE_W;
   if (descriptor->image_view) {
      const VkComponentMapping *view = &descriptor->image_view->vk.swizzle;
      unsigned char requested[4] = {rd_view_swizzle(view->r), rd_view_swizzle(view->g),
                                    rd_view_swizzle(view->b), rd_view_swizzle(view->a)};
      unsigned char native_swizzle[4];
      memcpy(native_swizzle, swizzle, sizeof(native_swizzle));
      util_format_compose_swizzles(native_swizzle, requested, swizzle);
   }
   out->format_word = A5XX_TEX_CONST_0_FMT(out->format) |
      A5XX_TEX_CONST_0_SAMPLES(samples) |
      A5XX_TEX_CONST_0_SWIZ_X(rd_resource_swizzle(swizzle[0])) |
      A5XX_TEX_CONST_0_SWIZ_Y(rd_resource_swizzle(swizzle[1])) |
      A5XX_TEX_CONST_0_SWIZ_Z(rd_resource_swizzle(swizzle[2])) |
      A5XX_TEX_CONST_0_SWIZ_W(rd_resource_swizzle(swizzle[3])) |
      COND(vk_format_is_srgb(format), A5XX_TEX_CONST_0_SRGB);
   return out->format != TFMT5_NONE;
}

bool
rd_descriptor_ready(const struct rd_descriptor *descriptor)
{
   switch (descriptor->type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      return descriptor->has_sampler;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      if (!descriptor->has_sampler)
         return false;
      FALLTHROUGH;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
      struct rd_resource resource;
      return rd_make_resource(descriptor, &resource);
   }
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return descriptor->buffer && descriptor->buffer->memory &&
         descriptor->offset <= descriptor->buffer->vk.size &&
         descriptor->range <= descriptor->buffer->vk.size - descriptor->offset;
   default:
      return false;
   }
}

static void
rd_resource_packet(struct fd_ringbuffer *ring, uint32_t slot,
                    enum a4xx_state_block block, enum a4xx_state_type type,
                    uint32_t count, uint32_t dwords)
{
   OUT_PKT7(ring, CP_LOAD_STATE4, 3 + dwords);
   OUT_RING(ring, CP_LOAD_STATE4_0_DST_OFF(slot) |
      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
      CP_LOAD_STATE4_0_STATE_BLOCK(block) | CP_LOAD_STATE4_0_NUM_UNIT(count));
   OUT_RING(ring, CP_LOAD_STATE4_1_STATE_TYPE(type));
   OUT_RING(ring, 0);
}

static void
rd_emit_resource_texture(struct fd_ringbuffer *ring, uint32_t slot,
                          const struct rd_resource *resource)
{
   rd_resource_packet(ring, slot, SB4_CS_TEX, ST4_CONSTANTS, 1, 12);
   OUT_RING(ring, resource->format_word | A5XX_TEX_CONST_0_MIPLVLS(resource->mip_levels));
   OUT_RING(ring, A5XX_TEX_CONST_1_WIDTH(resource->width) |
                  A5XX_TEX_CONST_1_HEIGHT(resource->height));
   OUT_RING(ring, COND(resource->buffer, A5XX_TEX_CONST_2_BUFFER) |
      A5XX_TEX_CONST_2_TYPE(resource->type) | A5XX_TEX_CONST_2_PITCHALIGN(resource->pitchalign) |
      A5XX_TEX_CONST_2_PITCH(resource->row_pitch));
   uint32_t texture_pitch = resource->image_size_only ||
      ((resource->type == A5XX_TEX_1D || resource->type == A5XX_TEX_2D) && resource->depth == 1) ?
      0 : resource->array_pitch;
   OUT_RING(ring, A5XX_TEX_CONST_3_ARRAY_PITCH(texture_pitch) |
      (resource->type == A5XX_TEX_3D ? A5XX_TEX_CONST_3_MIN_LAYERSZ(MIN2(resource->min_layer_size, 0xf000u)) : 0));
   OUT_RELOC(ring, resource->bo, resource->offset,
      (uint64_t)A5XX_TEX_CONST_5_DEPTH(resource->depth) << 32, 0);
   if (debug_get_bool_option("RADISH_RESOURCE_DIAGNOSTICS", false)) {
      uint64_t address = fd_bo_get_iova(resource->bo) + resource->offset;
      uint32_t words[6] = {
         resource->format_word | A5XX_TEX_CONST_0_MIPLVLS(resource->mip_levels),
         A5XX_TEX_CONST_1_WIDTH(resource->width) | A5XX_TEX_CONST_1_HEIGHT(resource->height),
         COND(resource->buffer, A5XX_TEX_CONST_2_BUFFER) | A5XX_TEX_CONST_2_TYPE(resource->type) |
            A5XX_TEX_CONST_2_PITCHALIGN(resource->pitchalign) | A5XX_TEX_CONST_2_PITCH(resource->row_pitch),
         A5XX_TEX_CONST_3_ARRAY_PITCH(texture_pitch) | (resource->type == A5XX_TEX_3D ?
            A5XX_TEX_CONST_3_MIN_LAYERSZ(MIN2(resource->min_layer_size, 0xf000u)) : 0),
         address, (address >> 32) | A5XX_TEX_CONST_5_DEPTH(resource->depth),
      };
      fprintf(stderr, "RADISH_TEXTURE_DIAG slot=%u buffer=%u size_only=%u bo=%llx bo_size=%u offset=%llu address=%llx cpp=%u width=%u height=%u depth=%u row=%u array=%u words=%08x,%08x,%08x,%08x,%08x,%08x,00000000,00000000,00000000,00000000,00000000,00000000\n",
         slot, resource->buffer, resource->image_size_only,
         (unsigned long long)fd_bo_get_iova(resource->bo), fd_bo_size(resource->bo),
         (unsigned long long)resource->offset, (unsigned long long)address, resource->cpp,
         resource->width, resource->height, resource->depth, resource->row_pitch, texture_pitch,
         words[0], words[1], words[2], words[3], words[4], words[5]);
   }
   for (uint32_t i = 0; i < 6; i++)
      OUT_RING(ring, 0);
}

static enum a5xx_tex_fmt
rd_storage_resource_format(enum a5xx_tex_fmt format)
{
   switch (format) {
   case TFMT5_8_SNORM: return TFMT5_8_SINT;
   case TFMT5_8_8_SNORM: return TFMT5_8_8_SINT;
   case TFMT5_8_8_8_8_SNORM: return TFMT5_8_8_8_8_SINT;
   case TFMT5_16_SNORM: return TFMT5_16_SINT;
   case TFMT5_16_16_SNORM: return TFMT5_16_16_SINT;
   case TFMT5_16_16_16_16_SNORM: return TFMT5_16_16_16_16_SINT;
   default: return format;
   }
}

static void
rd_emit_resource_image(struct fd_ringbuffer *ring, uint32_t slot,
                        const struct rd_resource *resource)
{
   enum a5xx_tex_fmt format = rd_storage_resource_format(resource->format);
   rd_resource_packet(ring, slot, SB4_CS_SSBO, ST4_CONSTANTS, 1, 2);
   uint32_t width = resource->buffer ? (1u << 15) : resource->width;
   uint32_t height = resource->buffer ? resource->height + !!resource->width : resource->height;
   OUT_RING(ring, A5XX_SSBO_1_0_FMT(format) |
                  A5XX_SSBO_1_0_WIDTH(width));
   OUT_RING(ring, A5XX_SSBO_1_1_HEIGHT(height) |
                  A5XX_SSBO_1_1_DEPTH(resource->depth));
   if (debug_get_bool_option("RADISH_RESOURCE_DIAGNOSTICS", false)) {
      uint64_t address = fd_bo_get_iova(resource->bo) + resource->offset;
      fprintf(stderr, "RADISH_IMAGE_DIAG slot=%u buffer=%u bo=%llx bo_size=%u offset=%llu address=%llx constants=%08x,%08x address_words=%08x,%08x\n",
         slot, resource->buffer, (unsigned long long)fd_bo_get_iova(resource->bo), fd_bo_size(resource->bo),
         (unsigned long long)resource->offset, (unsigned long long)address,
         A5XX_SSBO_1_0_FMT(format) | A5XX_SSBO_1_0_WIDTH(width),
         A5XX_SSBO_1_1_HEIGHT(height) | A5XX_SSBO_1_1_DEPTH(resource->depth),
         (uint32_t)address, (uint32_t)(address >> 32));
   }
   rd_resource_packet(ring, slot, SB4_CS_SSBO, ST4_UBO, 1, 2);
   OUT_RELOC(ring, resource->bo, resource->offset, 0, 0);
}

static enum a5xx_tex_clamp
rd_sampler_wrap(VkSamplerAddressMode wrap)
{
   switch (wrap) {
   case VK_SAMPLER_ADDRESS_MODE_REPEAT: return A5XX_TEX_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return A5XX_TEX_MIRROR_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return A5XX_TEX_CLAMP_TO_BORDER;
   default: return A5XX_TEX_CLAMP_TO_EDGE;
   }
}

void
rd_sampler_descriptor(const struct vk_sampler_state *sampler, uint32_t words[4])
{
   uint32_t word0 = A5XX_TEX_SAMP_0_WRAP_S(A5XX_TEX_CLAMP_TO_EDGE) |
      A5XX_TEX_SAMP_0_WRAP_T(A5XX_TEX_CLAMP_TO_EDGE) |
      A5XX_TEX_SAMP_0_WRAP_R(A5XX_TEX_CLAMP_TO_EDGE);
   uint32_t word1 = 0;
   if (sampler) {
      word0 = COND(sampler->mipmap_mode == VK_SAMPLER_MIPMAP_MODE_LINEAR,
                   A5XX_TEX_SAMP_0_MIPFILTER_LINEAR_NEAR) |
         A5XX_TEX_SAMP_0_XY_MAG(sampler->mag_filter == VK_FILTER_LINEAR ?
                               A5XX_TEX_LINEAR : A5XX_TEX_NEAREST) |
         A5XX_TEX_SAMP_0_XY_MIN(sampler->min_filter == VK_FILTER_LINEAR ?
                               A5XX_TEX_LINEAR : A5XX_TEX_NEAREST) |
         A5XX_TEX_SAMP_0_WRAP_S(rd_sampler_wrap(sampler->address_mode_u)) |
         A5XX_TEX_SAMP_0_WRAP_T(rd_sampler_wrap(sampler->address_mode_v)) |
         A5XX_TEX_SAMP_0_WRAP_R(rd_sampler_wrap(sampler->address_mode_w)) |
         A5XX_TEX_SAMP_0_LOD_BIAS(CLAMP(sampler->mip_lod_bias, -16.0f, 15.99609375f));
      word1 = COND(sampler->compare_enable, A5XX_TEX_SAMP_1_COMPARE_FUNC((enum adreno_compare_func)sampler->compare_op)) |
         COND(sampler->unnormalized_coordinates, A5XX_TEX_SAMP_1_UNNORM_COORDS) |
         A5XX_TEX_SAMP_1_MIN_LOD(CLAMP(sampler->min_lod, 0.0f, 15.99609375f)) |
         A5XX_TEX_SAMP_1_MAX_LOD(CLAMP(sampler->max_lod, 0.0f, 15.99609375f));
   }
   words[0] = word0;
   words[1] = word1;
   words[2] = sampler ? A5XX_TEX_SAMP_2_BCOLOR_OFFSET(sampler->border_color) : 0;
   words[3] = 0;
}

void
rd_emit_shader_resources(struct rd_device *device, struct fd_ringbuffer *ring,
                          const struct rd_dispatch *dispatch)
{
   const struct rd_pipeline *pipeline = dispatch->pipeline;
   const struct ir3_shader_variant *v = pipeline->variant;
   const struct rd_nir_resource_usage *usage = &pipeline->resource_usage;
   uint32_t sampler_count = usage->sampler_count;
   if (sampler_count) {
      OUT_PKT4(ring, REG_A5XX_TPL1_TP_BORDER_COLOR_BASE_ADDR, 2);
      OUT_RELOC(ring, device->border_colors, device->border_colors_offset, 0, 0);
   }
   for (uint32_t i = 0; i < sampler_count; i++) {
      uint32_t words[4];
      rd_sampler_descriptor(NULL, words);
      if (i < 16 && (usage->samplers & (1u << i)))
         memcpy(words, dispatch->descriptors[pipeline->sampler_descriptors[i]].sampler,
                sizeof(words));
      rd_resource_packet(ring, i, SB4_CS_TEX, ST4_SHADER, 1, 4);
      if (debug_get_bool_option("RADISH_RESOURCE_DIAGNOSTICS", false))
         fprintf(stderr, "RADISH_SAMPLER_DIAG slot=%u words=%08x,%08x,%08x,%08x\n",
            i, words[0], words[1], words[2], words[3]);
      for (uint32_t j = 0; j < 4; j++)
         OUT_RING(ring, words[j]);
   }
   for (uint32_t i = 0; i < usage->texture_count; i++) {
      if (!(usage->textures & (1u << i)))
         continue;
      struct rd_resource resource;
      if (rd_make_resource(&dispatch->descriptors[pipeline->texture_descriptors[i]],
                            &resource)) {
         if (pipeline->sampling_mask & (1u << i))
            resource.image_size_only = true;
         rd_emit_resource_texture(ring, i, &resource);
      }
   }
   for (uint32_t i = 0; i < usage->image_count; i++) {
      if (!(usage->images & (1u << i)))
         continue;
      struct rd_resource resource;
      if (rd_make_resource(&dispatch->descriptors[pipeline->image_descriptors[i]],
                            &resource))
         rd_emit_resource_image(ring, v->num_ssbos + i, &resource);
   }
   const struct ir3_ibo_mapping *mapping = &v->image_mapping;
   for (uint32_t i = 0; i < mapping->num_tex; i++) {
      uint32_t resource_index = mapping->tex_to_image[i];
      uint32_t slot = resource_index & UAV_SSBO ?
         pipeline->buffer_descriptors[resource_index & ~UAV_SSBO] :
         pipeline->image_descriptors[resource_index];
      struct rd_resource resource;
      if (rd_make_resource(&dispatch->descriptors[slot], &resource))
         rd_emit_resource_texture(ring, mapping->tex_base + i, &resource);
   }
}

void
rd_emit_image_constants(struct fd_ringbuffer *ring,
                         const struct rd_dispatch *dispatch)
{
   const struct rd_pipeline *pipeline = dispatch->pipeline;
   const struct ir3_shader_variant *v = pipeline->variant;
   const struct ir3_const_state *state = ir3_const_state(v);
   if (!state->image_dims.count ||
       !ir3_const_can_upload(&state->allocs, IR3_CONST_ALLOC_IMAGE_DIMS, v->constlen))
      return;
   uint32_t values[128] = {0};
   assert(state->image_dims.count <= ARRAY_SIZE(values));
   for (uint32_t i = 0; i < 16; i++) {
      if (!(state->image_dims.mask & (1u << i)))
         continue;
      struct rd_resource resource;
      if (!rd_make_resource(&dispatch->descriptors[pipeline->image_descriptors[i]],
                             &resource))
         continue;
      uint32_t offset = state->image_dims.off[i];
      values[offset] = resource.cpp;
      const struct rd_descriptor *descriptor = &dispatch->descriptors[pipeline->image_descriptors[i]];
      values[offset + 1] = resource.buffer ? util_logbase2(resource.cpp) :
         descriptor->image_view->vk.view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY ?
         resource.array_pitch : resource.row_pitch;
      values[offset + 2] = resource.array_pitch;
      if (v->shader_options.nir_options.a5xx_image_size_consts) {
         values[offset + 3] = resource.buffer ?
            descriptor->buffer_view->vk.elements : resource.width;
         values[offset + 4] = resource.height;
         values[offset + 5] = !resource.buffer &&
            (descriptor->image_view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
             descriptor->image_view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ?
            descriptor->image_view->vk.layer_count / 6 : resource.depth;
      }
   }
   for (uint32_t i = 0; i < IR3_MAX_SHADER_BUFFERS; i++) {
      if (!(state->ssbo_size_mask & (1u << i)))
         continue;
      assert(i < pipeline->buffer_count);
      bool metadata = dispatch->sampling_addresses && i == pipeline->sampling_buffer;
      const struct rd_descriptor *descriptor = metadata ? NULL :
         &dispatch->descriptors[pipeline->buffer_descriptors[i]];
      values[state->ssbo_size_offset[i]] = metadata ? 128 :
         descriptor->buffer ? descriptor->range : 0;
   }
   uint32_t offset = state->allocs.consts[IR3_CONST_ALLOC_IMAGE_DIMS].offset_vec4;
   uint32_t count = MIN2(align(state->image_dims.count, 4), (v->constlen - offset) * 4);
   rd_resource_packet(ring, offset, SB4_CS_SHADER, ST4_CONSTANTS, count / 4, count);
   for (uint32_t i = 0; i < count; i++)
      OUT_RING(ring, values[i]);
}
