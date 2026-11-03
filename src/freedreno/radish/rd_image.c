#include "rd_private.h"
#include "vk_format.h"
#include "vk_format_info.h"
#include "vk_util.h"
#include "util/format/u_format.h"
#include <limits.h>
#include <math.h>

static uint32_t
rd_image_texel_size(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return vk_format_get_blocksize(format);
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   case VK_FORMAT_A2R10G10B10_UINT_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
      return vk_format_get_blocksize(format);
   case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   case VK_FORMAT_B8G8R8A8_SNORM:
   case VK_FORMAT_B8G8R8A8_UINT:
   case VK_FORMAT_B8G8R8A8_SINT:
      return vk_format_get_blocksize(format);
   case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
   case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
   case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
   case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
   case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
   case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
   case VK_FORMAT_EAC_R11_UNORM_BLOCK:
   case VK_FORMAT_EAC_R11_SNORM_BLOCK:
   case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
   case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
      return vk_format_get_blocksize(format);
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_D32_SFLOAT:
   case VK_FORMAT_S8_UINT:
   case VK_FORMAT_D24_UNORM_S8_UINT:
      return vk_format_get_blocksize(format);
   default:
      return 0;
   }
}

static bool
rd_image_color_attachment_supported(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   case VK_FORMAT_A2R10G10B10_UINT_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return true;
   default:
      return false;
   }
}

static bool
rd_image_color_blend_supported(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return true;
   default:
      return false;
   }
}

static bool
rd_image_required_blit_src(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_D32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static bool
rd_image_required_blit_dst(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static bool
rd_image_required_vertex_buffer(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
   case VK_FORMAT_R32G32B32_UINT:
   case VK_FORMAT_R32G32B32_SINT:
   case VK_FORMAT_R32G32B32_SFLOAT:
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static VkImageUsageFlags2KHR
rd_image_shader_usage(VkFormat format)
{
   if (!rd_image_texel_size(format) || !rd_shader_format_supported(format))
      return 0;
   VkImageUsageFlags2KHR usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   if (rd_image_color_attachment_supported(format))
      usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
       format == VK_FORMAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT)
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if (rd_storage_shader_format_supported(format))
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;
   return usage;
}

static bool
rd_image_parameters(VkFormat format, VkImageType type, VkImageTiling tiling,
                    VkImageUsageFlags2KHR usage, VkImageCreateFlags2KHR flags)
{
   if (!rd_image_texel_size(format))
      return false;
   if (vk_format_is_depth_or_stencil(format) && type != VK_IMAGE_TYPE_2D)
      return false;
   if (vk_format_is_compressed(format) &&
       (type != VK_IMAGE_TYPE_2D || tiling != VK_IMAGE_TILING_OPTIMAL))
      return false;
   VkImageUsageFlags2KHR shader_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if ((usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) &&
       tiling != VK_IMAGE_TILING_OPTIMAL)
      return false;
   VkImageUsageFlags2KHR supported = rd_image_shader_usage(format);
   if ((flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) &&
       (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
       !vk_format_is_depth_or_stencil(format)) {
      const struct vk_format_class_info *class = vk_format_get_class_info(format);
      for (uint32_t i = 0; i < class->format_count; i++) {
         if (!vk_format_is_depth_or_stencil(class->formats[i]))
            supported |= rd_image_shader_usage(class->formats[i]);
      }
      if ((flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) && vk_format_is_compressed(format)) {
         VkFormat uncompressed = vk_format_get_blocksize(format) == 8 ?
            VK_FORMAT_R32G32_UINT : VK_FORMAT_R32G32B32A32_UINT;
         const struct vk_format_class_info *block = vk_format_get_class_info(uncompressed);
         for (uint32_t i = 0; i < block->format_count; i++)
            supported |= rd_image_shader_usage(block->formats[i]);
      }
   }
   if (usage & shader_usage & ~supported)
      return false;
   if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) {
      VkImageUsageFlags2KHR attachments = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
      if (!(usage & attachments) ||
          (usage & ~(attachments | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)))
         return false;
   }
   return rd_image_texel_size(format) &&
          (type == VK_IMAGE_TYPE_1D || type == VK_IMAGE_TYPE_2D || type == VK_IMAGE_TYPE_3D) &&
          (tiling == VK_IMAGE_TILING_LINEAR || tiling == VK_IMAGE_TILING_OPTIMAL) &&
          usage && !(usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | shader_usage)) &&
          !(flags & ~(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                       VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT |
                       VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
                       VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT)) &&
          (!(flags & VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT) ||
           (type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL)) &&
          (!(flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) || type == VK_IMAGE_TYPE_2D) &&
          (!(flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) ||
           (vk_format_is_compressed(format) && (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT))) &&
          (!(flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) || type == VK_IMAGE_TYPE_3D);
}

static VkSampleCountFlags
rd_image_sample_counts(VkFormat format, VkImageType type, VkImageTiling tiling,
                       VkImageUsageFlags2KHR usage, VkImageCreateFlags2KHR flags)
{
   VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT;
   if ((rd_image_shader_usage(format) & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !vk_format_is_compressed(format) && type == VK_IMAGE_TYPE_2D && tiling == VK_IMAGE_TILING_OPTIMAL &&
       !(usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
       !(flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))
      samples |= VK_SAMPLE_COUNT_4_BIT;
   return samples;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice handle, VkFormat format,
                                      VkFormatProperties2 *out)
{
   VkFormatFeatureFlags flags = rd_image_texel_size(format) ?
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT : 0;
   VkFormatFeatureFlags buffer = rd_image_required_vertex_buffer(format) ?
      VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT : 0;
   if (rd_shader_format_supported(format)) {
      if (rd_image_texel_size(format))
         flags |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
      if (rd_image_texel_size(format) && rd_shader_format_filterable(format)) {
         flags |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
         if (vk_format_is_compressed(format))
            flags |= VK_FORMAT_FEATURE_BLIT_SRC_BIT;
      }
      if (!vk_format_is_compressed(format) && !vk_format_is_srgb(format) && !vk_format_is_depth_or_stencil(format))
         buffer |= VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;
      if (rd_storage_shader_format_supported(format)) {
         flags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
         buffer |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
      }
      if (format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT) {
         flags |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
         buffer |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
      }
   }
   VkImageUsageFlags2KHR usage = rd_image_shader_usage(format);
   VkFormatFeatureFlags attachment = 0;
   if (rd_image_required_blit_src(format))
      attachment |= VK_FORMAT_FEATURE_BLIT_SRC_BIT;
   if (rd_image_required_blit_dst(format))
      attachment |= VK_FORMAT_FEATURE_BLIT_DST_BIT;
   if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
      attachment |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
      if (rd_image_color_blend_supported(format))
         attachment |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
   }
   if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
      attachment |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
   out->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = vk_format_is_compressed(format) ? 0 : flags, .optimalTilingFeatures = flags | attachment,
      .bufferFeatures = buffer,
   };
   VkFormatFeatureFlags2 flags2 = flags;
   if (vk_format_has_depth(format) && (flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
      flags2 |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
   vk_foreach_struct(sType, next, out->pNext) {
      if (sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3) {
         VkFormatProperties3 *properties = (VkFormatProperties3 *)next;
         properties->linearTilingFeatures = vk_format_is_compressed(format) ? 0 : flags2;
         properties->optimalTilingFeatures = flags2 | attachment;
         properties->bufferFeatures = buffer;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice handle,
   const VkPhysicalDeviceImageFormatInfo2 *info, VkImageFormatProperties2 *out)
{
   out->imageFormatProperties = (VkImageFormatProperties){0};
   const VkPhysicalDeviceExternalImageFormatInfo *external =
      vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   if (!rd_image_parameters(info->format, info->type, info->tiling, info->usage, info->flags) ||
       (external && external->handleType))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   uint32_t side = info->type == VK_IMAGE_TYPE_3D ? 256 : 4096;
   out->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = {side, info->type == VK_IMAGE_TYPE_1D ? 1 : side,
                    info->type == VK_IMAGE_TYPE_3D ? side : 1},
      .maxMipLevels = info->type == VK_IMAGE_TYPE_3D ? 9 : RD_IMAGE_MAX_LEVELS,
      .maxArrayLayers = info->type == VK_IMAGE_TYPE_3D ? 1 : 256,
      .sampleCounts = rd_image_sample_counts(info->format, info->type, info->tiling, info->usage, info->flags),
      .maxResourceSize = RD_IMAGE_MAX_SIZE,
   };
   if (info->flags & VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT) {
      out->imageFormatProperties.maxMipLevels = 1;
      out->imageFormatProperties.maxArrayLayers = 1;
   }
   vk_foreach_struct(sType, next, out->pNext) {
      if (sType == VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES)
         ((VkExternalImageFormatProperties *)next)->externalMemoryProperties =
            (VkExternalMemoryProperties){0};
   }
   return VK_SUCCESS;
}

static bool
rd_image_layout(struct rd_image *image)
{
   VkDeviceSize size = 0;
   bool shader = image->vk.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
   VkDeviceSize limit = RD_IMAGE_MAX_SIZE;
   if (shader) {
      const struct fdl_image_params parameters = {
         .format = vk_format_to_pipe_format(image->vk.format), .nr_samples = image->vk.samples,
         .width0 = image->vk.extent.width, .height0 = image->vk.extent.height,
         .depth0 = image->vk.extent.depth, .mip_levels = image->vk.mip_levels,
         .array_size = image->vk.array_layers, .is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D,
      };
      fdl5_layout_image(&image->native_layout, &parameters);
      image->native_layout.mip_levels = image->vk.mip_levels;
      if (!image->native_layout.size || image->native_layout.size > limit)
         return false;
      image->size = image->native_layout.size;
      for (uint32_t level = 0; level < image->vk.mip_levels; level++) {
         VkExtent3D extent = vk_image_mip_level_extent(&image->vk, level);
         struct rd_image_level *layout = &image->levels[level];
         layout->offset = fdl_surface_offset(&image->native_layout, level, 0);
         layout->row_pitch = fdl_pitch(&image->native_layout, level);
         layout->depth_pitch = image->native_layout.slices[level].size0;
         layout->size = layout->depth_pitch * extent.depth;
         layout->array_pitch = image->native_layout.layer_first ?
            image->native_layout.layer_size : layout->size;
         if (!layout->row_pitch || !layout->array_pitch || layout->row_pitch > UINT32_MAX ||
             layout->depth_pitch > UINT32_MAX || layout->array_pitch > UINT32_MAX ||
             layout->offset > image->size || layout->size > image->size - layout->offset ||
             (image->vk.array_layers - 1) >
                (image->size - layout->offset - layout->size) / layout->array_pitch)
            return false;
      }
      return true;
   }
   for (uint32_t level = 0; level < image->vk.mip_levels; level++) {
      VkExtent3D extent = vk_image_mip_level_extent(&image->vk, level);
      struct rd_image_level *layout = &image->levels[level];
      layout->offset = size;
      VkExtent3D elements = vk_image_extent_to_elements(&image->vk, extent);
      layout->row_pitch = align64((uint64_t)elements.width * image->texel_size * image->vk.samples, 64);
      layout->depth_pitch = layout->row_pitch * elements.height;
      if (shader)
         layout->depth_pitch = align64(layout->depth_pitch, 4096);
      layout->array_pitch = layout->depth_pitch * extent.depth;
      layout->size = layout->array_pitch;
      VkDeviceSize level_size = layout->array_pitch * image->vk.array_layers;
      if (level_size > limit - size)
         return false;
      size += level_size;
   }
   image->size = size;
   return true;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateImage(VkDevice handle, const VkImageCreateInfo *info,
               const VkAllocationCallbacks *allocator, VkImage *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   *out = VK_NULL_HANDLE;
   VkImageCreateFlags2KHR flags = vk_image_create_flags(info);
   if (!rd_image_parameters(info->format, info->imageType, info->tiling,
                            vk_image_usage_flags(info), flags) ||
       (info->samples != VK_SAMPLE_COUNT_1_BIT && info->samples != VK_SAMPLE_COUNT_4_BIT) ||
       !(info->samples & rd_image_sample_counts(info->format, info->imageType, info->tiling,
                                               vk_image_usage_flags(info), flags)) ||
       (info->samples != VK_SAMPLE_COUNT_1_BIT && info->mipLevels != 1) || !info->mipLevels ||
       info->mipLevels > RD_IMAGE_MAX_LEVELS || !info->arrayLayers ||
       info->arrayLayers > 256 || !info->extent.width || !info->extent.height ||
       !info->extent.depth ||
       (info->initialLayout != VK_IMAGE_LAYOUT_UNDEFINED &&
        info->initialLayout != VK_IMAGE_LAYOUT_PREINITIALIZED))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((flags & VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT) &&
       (info->mipLevels != 1 || info->arrayLayers != 1))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   uint32_t side = info->imageType == VK_IMAGE_TYPE_3D ? 256 : 4096;
   if (info->extent.width > side || info->extent.height > side || info->extent.depth > side ||
       (info->imageType == VK_IMAGE_TYPE_1D && (info->extent.height != 1 || info->extent.depth != 1)) ||
       (info->imageType == VK_IMAGE_TYPE_2D && info->extent.depth != 1) ||
       (info->imageType == VK_IMAGE_TYPE_3D && info->arrayLayers != 1) ||
       info->mipLevels > util_logbase2(MAX3(info->extent.width, info->extent.height, info->extent.depth)) + 1 ||
       ((flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
        (info->extent.width != info->extent.height || info->arrayLayers < 6)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   const VkExternalMemoryImageCreateInfo *external =
      vk_find_struct_const(info->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   if (external && external->handleTypes)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct rd_image *image = vk_image_create(&device->vk, info, allocator, sizeof(*image));
   if (!image)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   image->texel_size = rd_image_texel_size(info->format);
   if (!rd_image_layout(image)) {
      vk_image_destroy(&device->vk, allocator, &image->vk);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   *out = rd_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyImage(VkDevice handle, VkImage image_handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_image, image, image_handle);
   if (image)
      vk_image_destroy(&device->vk, allocator, &image->vk);
}

VKAPI_ATTR void VKAPI_CALL
rd_GetImageMemoryRequirements2(VkDevice handle, const VkImageMemoryRequirementsInfo2 *info,
                              VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(rd_image, image, info->image);
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = image->size, .alignment = 64, .memoryTypeBits = 1,
   };
   vk_foreach_struct(sType, next, out->pNext) {
      if (sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
         VkMemoryDedicatedRequirements *dedicated = (VkMemoryDedicatedRequirements *)next;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_GetImageMemoryRequirements(VkDevice handle, VkImage image, VkMemoryRequirements *out)
{
   VkImageMemoryRequirementsInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = image,
   };
   VkMemoryRequirements2 result = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
   rd_GetImageMemoryRequirements2(handle, &info, &result);
   *out = result.memoryRequirements;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_BindImageMemory2(VkDevice handle, uint32_t count, const VkBindImageMemoryInfo *infos)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(rd_image, image, infos[i].image);
      VK_FROM_HANDLE(rd_memory, memory, infos[i].memory);
      const VkBindImageMemoryDeviceGroupInfo *group =
         vk_find_struct_const(infos[i].pNext, BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO);
      if (group) {
         if (group->deviceIndexCount > 1 || group->splitInstanceBindRegionCount > 1 ||
             (group->deviceIndexCount && group->splitInstanceBindRegionCount) ||
             (group->deviceIndexCount && (!group->pDeviceIndices || group->pDeviceIndices[0])))
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         if (group->splitInstanceBindRegionCount) {
            if (!image || !(image->vk.create_flags & VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT) ||
                !group->pSplitInstanceBindRegions)
               return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            const VkRect2D *region = group->pSplitInstanceBindRegions;
            if (region->offset.x || region->offset.y ||
                region->extent.width != image->vk.extent.width ||
                region->extent.height != image->vk.extent.height)
               return VK_ERROR_OUT_OF_DEVICE_MEMORY;
         }
      }
      if (!image || !memory || image->vk.base.device != &device->vk ||
          memory->vk.base.device != &device->vk || image->memory || (infos[i].memoryOffset & 63) ||
          infos[i].memoryOffset > memory->vk.size || image->size > memory->vk.size - infos[i].memoryOffset)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(rd_image, image, infos[i].image);
      image->memory = rd_memory_from_handle(infos[i].memory);
      image->memory_offset = infos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_BindImageMemory(VkDevice handle, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
   VkBindImageMemoryInfo info = {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                               .image = image, .memory = memory, .memoryOffset = offset};
   return rd_BindImageMemory2(handle, 1, &info);
}

VKAPI_ATTR void VKAPI_CALL
rd_GetImageSparseMemoryRequirements(VkDevice handle, VkImage image, uint32_t *count,
                                   VkSparseImageMemoryRequirements *out)
{
   *count = 0;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetImageSparseMemoryRequirements2(VkDevice handle, const VkImageSparseMemoryRequirementsInfo2 *info,
                                    uint32_t *count, VkSparseImageMemoryRequirements2 *out)
{
   *count = 0;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetImageSubresourceLayout(VkDevice handle, VkImage image_handle,
                            const VkImageSubresource *subresource, VkSubresourceLayout *out)
{
   VK_FROM_HANDLE(rd_image, image, image_handle);
   *out = (VkSubresourceLayout){0};
   if (!subresource->aspectMask || (subresource->aspectMask & ~vk_format_aspects(image->vk.format)) ||
       (subresource->aspectMask & (subresource->aspectMask - 1)) ||
       subresource->mipLevel >= image->vk.mip_levels || subresource->arrayLayer >= image->vk.array_layers)
      return;
   const struct rd_image_level *level = &image->levels[subresource->mipLevel];
   *out = (VkSubresourceLayout) {
      .offset = level->offset + subresource->arrayLayer * level->array_pitch,
      .size = level->size, .rowPitch = level->row_pitch,
      .arrayPitch = level->array_pitch, .depthPitch = level->depth_pitch,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateImageView(VkDevice handle, const VkImageViewCreateInfo *info,
                   const VkAllocationCallbacks *allocator, VkImageView *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_image, image, info->image);
   *out = VK_NULL_HANDLE;
   VkImageUsageFlags2KHR view_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   const VkImageSubresourceRange *r = &info->subresourceRange;
   bool block_view = image && (image->vk.create_flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) &&
      !vk_format_is_compressed(info->format) &&
      vk_format_get_blocksize(info->format) == vk_format_get_blocksize(image->vk.format);
   bool format_compatible = image && (block_view || info->format == image->vk.format ||
      ((image->vk.create_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
       !vk_format_is_depth_or_stencil(image->vk.format) &&
       !vk_format_is_depth_or_stencil(info->format) &&
       vk_format_get_class_info(info->format) == vk_format_get_class_info(image->vk.format)));
   if (!image || info->flags || !format_compatible ||
       !(image->vk.usage & view_usage) || !r->aspectMask ||
       (r->aspectMask & ~vk_format_aspects(image->vk.format)) ||
       r->baseMipLevel >= image->vk.mip_levels)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   bool slice_view = image->vk.image_type == VK_IMAGE_TYPE_3D &&
      (image->vk.create_flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) &&
      (info->viewType == VK_IMAGE_VIEW_TYPE_2D || info->viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
   uint32_t max_layers = slice_view ?
      vk_image_mip_level_extent(&image->vk, r->baseMipLevel).depth : image->vk.array_layers;
   if (r->baseArrayLayer >= max_layers)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   uint32_t levels = vk_image_subresource_level_count(&image->vk, r);
   uint32_t layers = r->layerCount == VK_REMAINING_ARRAY_LAYERS ?
      max_layers - r->baseArrayLayer : r->layerCount;
   if (!levels || levels > image->vk.mip_levels - r->baseMipLevel || !layers ||
       layers > max_layers - r->baseArrayLayer || (slice_view && levels != 1) ||
       (block_view && (levels != 1 || layers != 1)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   bool compatible = (image->vk.image_type == VK_IMAGE_TYPE_1D &&
      (info->viewType == VK_IMAGE_VIEW_TYPE_1D || info->viewType == VK_IMAGE_VIEW_TYPE_1D_ARRAY)) ||
      (image->vk.image_type == VK_IMAGE_TYPE_2D &&
      (info->viewType == VK_IMAGE_VIEW_TYPE_2D || info->viewType == VK_IMAGE_VIEW_TYPE_2D_ARRAY)) ||
      (image->vk.image_type == VK_IMAGE_TYPE_3D && info->viewType == VK_IMAGE_VIEW_TYPE_3D) ||
      slice_view;
   if ((info->viewType == VK_IMAGE_VIEW_TYPE_CUBE || info->viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) &&
       image->vk.image_type == VK_IMAGE_TYPE_2D &&
       (image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       (info->viewType == VK_IMAGE_VIEW_TYPE_CUBE ? layers == 6 : !(layers % 6)))
      compatible = true;
   const VkImageViewUsageCreateInfo *usage_info =
      vk_find_struct_const(info->pNext, IMAGE_VIEW_USAGE_CREATE_INFO);
   VkImageUsageFlags2KHR usage = usage_info ? usage_info->usage : image->vk.usage;
   if ((usage & ~image->vk.usage) || !rd_shader_format_supported(info->format) ||
       ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !rd_storage_shader_format_supported(info->format)) ||
       (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) &
        ~rd_image_shader_usage(info->format)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if (!compatible || ((info->viewType == VK_IMAGE_VIEW_TYPE_1D || info->viewType == VK_IMAGE_VIEW_TYPE_2D) && layers != 1))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   struct rd_image_view *view = vk_image_view_create(&device->vk, info, allocator, sizeof(*view));
   if (!view)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *out = rd_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyImageView(VkDevice handle, VkImageView view_handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_image_view, view, view_handle);
   if (view)
      vk_image_view_destroy(&device->vk, allocator, &view->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateBufferView(VkDevice handle, const VkBufferViewCreateInfo *info,
                    const VkAllocationCallbacks *allocator, VkBufferView *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_buffer, buffer, info->buffer);
   *out = VK_NULL_HANDLE;
   uint32_t size = vk_format_get_blocksize(info->format);
   if (!buffer || info->flags || !rd_shader_format_supported(info->format) ||
       vk_format_is_compressed(info->format) || vk_format_is_srgb(info->format) || vk_format_is_depth_or_stencil(info->format) ||
       !size || info->offset >= buffer->vk.size ||
       (info->offset & 31) || !(buffer->vk.usage &
       (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   VkDeviceSize range = info->range == VK_WHOLE_SIZE ? buffer->vk.size - info->offset : info->range;
   if (range > buffer->vk.size - info->offset ||
       (info->range != VK_WHOLE_SIZE && (!range || range % size)) ||
       range / size > (1u << 27))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   range -= range % size;
   struct rd_buffer_view *view = vk_buffer_view_create(&device->vk, info, allocator, sizeof(*view));
   if (!view)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   view->vk.range = range;
   view->vk.elements = range / size;
   *out = rd_buffer_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyBufferView(VkDevice handle, VkBufferView view_handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_buffer_view, view, view_handle);
   if (view)
      vk_buffer_view_destroy(&device->vk, allocator, &view->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateSampler(VkDevice handle, const VkSamplerCreateInfo *info,
                  const VkAllocationCallbacks *allocator, VkSampler *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   *out = VK_NULL_HANDLE;
   if (info->flags || info->anisotropyEnable ||
       (info->magFilter != VK_FILTER_NEAREST && info->magFilter != VK_FILTER_LINEAR) ||
       (info->minFilter != VK_FILTER_NEAREST && info->minFilter != VK_FILTER_LINEAR) ||
       info->mipmapMode > VK_SAMPLER_MIPMAP_MODE_LINEAR ||
       info->addressModeU > VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
       info->addressModeV > VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
       info->addressModeW > VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
       info->borderColor > VK_BORDER_COLOR_INT_OPAQUE_WHITE ||
       info->compareOp > VK_COMPARE_OP_ALWAYS || info->minLod > info->maxLod ||
       fabsf(info->mipLodBias) > 16.0f ||
       !isfinite(info->minLod) || !isfinite(info->maxLod) || !isfinite(info->mipLodBias))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct rd_sampler *sampler = vk_sampler_create(&device->vk, info, allocator, sizeof(*sampler));
   if (!sampler)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   vk_sampler_state_init(&sampler->state, info);
   *out = rd_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroySampler(VkDevice handle, VkSampler sampler_handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_sampler, sampler, sampler_handle);
   if (sampler)
      vk_sampler_destroy(&device->vk, allocator, &sampler->vk);
}

static bool
rd_image_range(const struct rd_image *image, const VkImageSubresourceRange *range)
{
   if (!image || range->aspectMask != vk_format_aspects(image->vk.format) ||
       range->baseMipLevel >= image->vk.mip_levels || range->baseArrayLayer >= image->vk.array_layers)
      return false;
   uint32_t levels = vk_image_subresource_level_count(&image->vk, range);
   uint32_t layers = vk_image_subresource_layer_count(&image->vk, range);
   return levels && levels <= image->vk.mip_levels - range->baseMipLevel &&
          layers && layers <= image->vk.array_layers - range->baseArrayLayer;
}

bool
rd_image_barrier_valid(const VkImageMemoryBarrier2 *barrier)
{
   VK_FROM_HANDLE(rd_image, image, barrier->image);
   if (!rd_image_range(image, &barrier->subresourceRange))
      return false;
   switch (barrier->oldLayout) {
   case VK_IMAGE_LAYOUT_UNDEFINED:
   case VK_IMAGE_LAYOUT_PREINITIALIZED:
   case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
   case VK_IMAGE_LAYOUT_GENERAL:
   case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
   case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      break;
   default:
      return false;
   }
   if (barrier->newLayout != VK_IMAGE_LAYOUT_GENERAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
       barrier->newLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
      return false;
   if ((barrier->srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED && barrier->srcQueueFamilyIndex != 0) ||
       (barrier->dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED && barrier->dstQueueFamilyIndex != 0))
      return false;
   return ((barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL &&
            barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL &&
            barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL) ||
           (vk_format_is_depth_or_stencil(image->vk.format) &&
            (image->vk.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) &&
          (barrier->newLayout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
           (image->vk.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
          (barrier->newLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
           (image->vk.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))) &&
          (barrier->newLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
           (image->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) &&
          (barrier->newLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) &&
          (barrier->newLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL || (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT));
}

static bool
rd_image_box(struct rd_image *image, const VkImageSubresourceLayers *subresource,
             VkOffset3D offset, VkExtent3D extent, VkDeviceSize *base,
             VkDeviceSize *row, VkDeviceSize *slice, uint32_t *slices)
{
   if (!image || !subresource->aspectMask ||
       (subresource->aspectMask & ~vk_format_aspects(image->vk.format)) ||
       subresource->mipLevel >= image->vk.mip_levels || !subresource->layerCount ||
       subresource->baseArrayLayer >= image->vk.array_layers ||
       subresource->layerCount > image->vk.array_layers - subresource->baseArrayLayer ||
       offset.x < 0 || offset.y < 0 || offset.z < 0 || !extent.width || !extent.height || !extent.depth)
      return false;
   VkExtent3D limit = vk_image_mip_level_extent(&image->vk, subresource->mipLevel);
   uint32_t bw = vk_format_get_blockwidth(image->vk.format);
   uint32_t bh = vk_format_get_blockheight(image->vk.format);
   VkExtent3D elements = vk_image_extent_to_elements(&image->vk, limit);
   uint32_t width = elements.width * bw;
   uint32_t height = elements.height * bh;
   if ((uint32_t)offset.x > width || extent.width > width - offset.x ||
       (uint32_t)offset.y > height || extent.height > height - offset.y ||
       (uint32_t)offset.z > limit.depth || extent.depth > limit.depth - offset.z)
      return false;
   if (offset.x % bw || offset.y % bh ||
       (extent.width % bw && extent.width != limit.width - offset.x) ||
       (extent.height % bh && extent.height != limit.height - offset.y))
      return false;
   VkOffset3D element_offset = vk_image_offset_to_elements(&image->vk, offset);
   const struct rd_image_level *level = &image->levels[subresource->mipLevel];
   *base = level->offset + subresource->baseArrayLayer * level->array_pitch +
           element_offset.z * level->depth_pitch + element_offset.y * level->row_pitch +
           (uint64_t)element_offset.x * image->texel_size * image->vk.samples;
   *row = level->row_pitch;
   *slice = image->vk.image_type == VK_IMAGE_TYPE_3D ? level->depth_pitch : level->array_pitch;
   *slices = image->vk.image_type == VK_IMAGE_TYPE_3D ? extent.depth : subresource->layerCount;
   return true;
}

static bool
rd_image_copy_layout(VkImageLayout layout, bool source)
{
   return layout == VK_IMAGE_LAYOUT_GENERAL || layout ==
      (source ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
}

static bool
rd_image_command_valid(struct rd_command_buffer *command, bool valid)
{
   if (!valid)
      vk_command_buffer_set_error(&command->vk, VK_ERROR_UNKNOWN);
   return valid;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdCopyImage(VkCommandBuffer handle, VkImage src_handle, VkImageLayout src_layout,
                VkImage dst_handle, VkImageLayout dst_layout, uint32_t count, const VkImageCopy *regions)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_image, src, src_handle);
   VK_FROM_HANDLE(rd_image, dst, dst_handle);
   if (!rd_image_command_valid(command, src && dst && src->texel_size == dst->texel_size &&
       src->vk.samples == dst->vk.samples &&
       (!(vk_format_is_depth_or_stencil(src->vk.format) || vk_format_is_depth_or_stencil(dst->vk.format)) ||
        src->vk.format == dst->vk.format) &&
       (src->vk.image_type == dst->vk.image_type ||
        (src->vk.image_type == VK_IMAGE_TYPE_2D && dst->vk.image_type == VK_IMAGE_TYPE_3D) ||
        (src->vk.image_type == VK_IMAGE_TYPE_3D && dst->vk.image_type == VK_IMAGE_TYPE_2D)) &&
       (src->vk.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
       (dst->vk.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
       rd_image_copy_layout(src_layout, true) && rd_image_copy_layout(dst_layout, false)))
      return;
   for (uint32_t i = 0; i < count; i++) {
      struct rd_image_transfer transfer = {.src_image = src, .dst_image = dst};
      if (!rd_image_command_valid(command,
          regions[i].srcSubresource.aspectMask == regions[i].dstSubresource.aspectMask))
         return;
      if (src->vk.format == VK_FORMAT_D24_UNORM_S8_UINT) {
         if (regions[i].srcSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT)
            transfer.copy_mask = 0x00ffffffu;
         else if (regions[i].srcSubresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)
            transfer.copy_mask = 0xff000000u;
      }
      VkExtent3D src_extent = regions[i].extent;
      VkExtent3D dst_extent = regions[i].extent;
      if (src->vk.image_type != dst->vk.image_type) {
         if (src->vk.image_type == VK_IMAGE_TYPE_2D)
            src_extent.depth = 1;
         else
            dst_extent.depth = 1;
      }
      VkExtent3D elements = vk_image_extent_to_elements(&src->vk, src_extent);
      dst_extent.width = elements.width * vk_format_get_blockwidth(dst->vk.format);
      dst_extent.height = elements.height * vk_format_get_blockheight(dst->vk.format);
      uint32_t src_slices, dst_slices;
      if (!rd_image_command_valid(command,
          rd_image_box(src, &regions[i].srcSubresource, regions[i].srcOffset, src_extent,
             &transfer.src_offset, &transfer.src_row_pitch, &transfer.src_slice_pitch, &src_slices) &&
          rd_image_box(dst, &regions[i].dstSubresource, regions[i].dstOffset, dst_extent,
             &transfer.dst_offset, &transfer.dst_row_pitch, &transfer.dst_slice_pitch, &dst_slices) && src_slices == dst_slices))
         return;
      transfer.row_bytes = (uint64_t)elements.width * src->texel_size * src->vk.samples;
      transfer.rows = elements.height;
      transfer.slices = src_slices;
      struct rd_command *recorded = rd_command_append(command, RD_CMD_IMAGE_COPY);
      if (!recorded)
         return;
      recorded->image_transfer = transfer;
   }
}

static void
rd_record_buffer_image(struct rd_command_buffer *command, struct rd_buffer *buffer,
                       struct rd_image *image, VkImageLayout layout, bool to_image,
                       uint32_t count, const VkBufferImageCopy *regions)
{
   if (!rd_image_command_valid(command, buffer && image && image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
       (buffer->vk.usage & (to_image ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT)) &&
       (image->vk.usage & (to_image ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) &&
       rd_image_copy_layout(layout, !to_image)))
      return;
   for (uint32_t i = 0; i < count; i++) {
      const VkBufferImageCopy *region = &regions[i];
      /* Depth/stencil: single-aspect formats copy texels as they are stored.
       * The depth aspect of D24S8 is stored and copied as X8_D24 words; on
       * the way in, the stencil byte of each word is preserved. The stencil
       * aspect of D24S8 would need repacking and is not supported.
       */
      uint32_t copy_mask = 0;
      bool stencil_lane = false;
      if (image->vk.format == VK_FORMAT_D24_UNORM_S8_UINT) {
         if (region->imageSubresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)
            stencil_lane = true;
         else if (!rd_image_command_valid(command,
             region->imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT))
            return;
         copy_mask = to_image && !stencil_lane ? 0x00ffffffu : 0;
      }
      VkDeviceSize image_offset, image_row, image_slice;
      uint32_t slices;
      if (!rd_image_command_valid(command,
          rd_image_box(image, &region->imageSubresource, region->imageOffset, region->imageExtent,
             &image_offset, &image_row, &image_slice, &slices)))
         return;
      uint64_t row_length = region->bufferRowLength ? region->bufferRowLength : region->imageExtent.width;
      uint64_t height = region->bufferImageHeight ? region->bufferImageHeight : region->imageExtent.height;
      uint32_t bw = vk_format_get_blockwidth(image->vk.format);
      uint32_t bh = vk_format_get_blockheight(image->vk.format);
      uint64_t row = DIV_ROUND_UP(row_length, bw) * image->texel_size;
      if (!rd_image_command_valid(command, row_length >= region->imageExtent.width &&
          height >= region->imageExtent.height && row <= RD_IMAGE_MAX_SIZE &&
          (!region->bufferRowLength || !(region->bufferRowLength % bw)) &&
          (!region->bufferImageHeight || !(region->bufferImageHeight % bh)) &&
          DIV_ROUND_UP(height, bh) <= RD_IMAGE_MAX_SIZE / row &&
          !(region->bufferOffset % image->texel_size)))
         return;
      VkBufferImageCopy2 region2 = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
         .bufferOffset = region->bufferOffset,
         .bufferRowLength = region->bufferRowLength,
         .bufferImageHeight = region->bufferImageHeight,
         .imageSubresource = region->imageSubresource,
         .imageOffset = region->imageOffset,
         .imageExtent = region->imageExtent,
      };
      struct vk_image_buffer_layout buffer_layout = vk_image_buffer_copy_layout(&image->vk, &region2);
      VkExtent3D elements = vk_image_extent_to_elements(&image->vk, region->imageExtent);
      row = buffer_layout.row_stride_B;
      uint64_t slice = buffer_layout.image_stride_B;
      uint64_t rows = elements.height;
      uint64_t row_bytes = (uint64_t)elements.width * buffer_layout.element_size_B;
      uint64_t end = vk_image_buffer_range(&image->vk, &buffer_layout, &elements, &region->imageSubresource);
      if (!rd_image_command_valid(command, region->bufferOffset <= buffer->vk.size &&
          end <= buffer->vk.size - region->bufferOffset))
         return;
      struct rd_image_transfer transfer = {
         .src_image = to_image ? NULL : image, .dst_image = to_image ? image : NULL,
         .src_buffer = to_image ? buffer : NULL, .dst_buffer = to_image ? NULL : buffer,
         .src_offset = to_image ? region->bufferOffset : image_offset,
         .dst_offset = to_image ? image_offset : region->bufferOffset,
         .src_row_pitch = to_image ? row : image_row, .dst_row_pitch = to_image ? image_row : row,
         .src_slice_pitch = to_image ? slice : image_slice, .dst_slice_pitch = to_image ? image_slice : slice,
         .row_bytes = row_bytes, .rows = rows, .slices = slices, .copy_mask = copy_mask,
      };
      if (stencil_lane) {
         /* Buffer: one stencil byte per texel. Image: byte 3 of each word. */
         transfer.lane_step = 1;
         transfer.src_step = to_image ? 1 : 4;
         transfer.dst_step = to_image ? 4 : 1;
         transfer.src_lane = to_image ? 0 : 3;
         transfer.dst_lane = to_image ? 3 : 0;
         transfer.row_bytes = elements.width;
      }
      struct rd_command *recorded = rd_command_append(command, RD_CMD_IMAGE_COPY);
      if (!recorded)
         return;
      recorded->image_transfer = transfer;
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdCopyBufferToImage(VkCommandBuffer handle, VkBuffer src, VkImage dst, VkImageLayout layout,
                       uint32_t count, const VkBufferImageCopy *regions)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   rd_record_buffer_image(command, rd_buffer_from_handle(src), rd_image_from_handle(dst), layout, true, count, regions);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdCopyImageToBuffer(VkCommandBuffer handle, VkImage src, VkImageLayout layout, VkBuffer dst,
                       uint32_t count, const VkBufferImageCopy *regions)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   rd_record_buffer_image(command, rd_buffer_from_handle(dst), rd_image_from_handle(src), layout, false, count, regions);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdClearColorImage(VkCommandBuffer handle, VkImage image_handle, VkImageLayout layout,
                      const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_image, image, image_handle);
   if (!rd_image_command_valid(command, image && !vk_format_is_depth_or_stencil(image->vk.format) &&
       !vk_format_is_compressed(image->vk.format) &&
       (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
       rd_image_copy_layout(layout, false)))
      return;
   uint8_t clear[16] = {0};
   util_format_pack_rgba(vk_format_to_pipe_format(image->vk.format), clear, color, 1);
   for (uint32_t i = 0; i < count; i++) {
      if (!rd_image_command_valid(command, rd_image_range(image, &ranges[i])))
         return;
      uint32_t levels = vk_image_subresource_level_count(&image->vk, &ranges[i]);
      uint32_t layers = vk_image_subresource_layer_count(&image->vk, &ranges[i]);
      for (uint32_t m = 0; m < levels; m++) {
         uint32_t level = ranges[i].baseMipLevel + m;
         VkExtent3D extent = vk_image_mip_level_extent(&image->vk, level);
         struct rd_image_transfer transfer = {
            .dst_image = image,
            .dst_offset = image->levels[level].offset + ranges[i].baseArrayLayer * image->levels[level].array_pitch,
            .dst_row_pitch = image->levels[level].row_pitch,
            .dst_slice_pitch = image->vk.image_type == VK_IMAGE_TYPE_3D ? image->levels[level].depth_pitch : image->levels[level].array_pitch,
            .row_bytes = (uint64_t)extent.width * image->texel_size * image->vk.samples,
            .rows = extent.height, .slices = image->vk.image_type == VK_IMAGE_TYPE_3D ? extent.depth : layers,
            .texel_size = image->texel_size,
         };
         memcpy(transfer.clear, clear, sizeof(clear));
         struct rd_command *recorded = rd_command_append(command, RD_CMD_IMAGE_CLEAR);
         if (!recorded)
            return;
         recorded->image_transfer = transfer;
      }
   }
}

static bool
rd_image_transfer_memory(struct rd_image *image, struct rd_buffer *buffer,
                         struct rd_memory **memory, VkDeviceSize *offset, VkDeviceSize *size)
{
   if (image) {
      *memory = image->memory;
      *offset = image->memory_offset;
      *size = image->size;
   } else if (buffer) {
      *memory = buffer->memory;
      *offset = buffer->offset;
      *size = buffer->vk.size;
   } else {
      return false;
   }
   return *memory && *offset <= (*memory)->vk.size && *size <= (*memory)->vk.size - *offset;
}

static bool
rd_image_transfer_bounds(VkDeviceSize offset, VkDeviceSize row, VkDeviceSize slice,
                         const struct rd_image_transfer *transfer, VkDeviceSize size,
                         uint32_t step, uint32_t lane)
{
   if (!transfer->rows || !transfer->slices || !transfer->row_bytes || offset > size)
      return false;
   uint64_t remaining = size - offset;
   uint64_t span = transfer->lane_step ?
      (transfer->row_bytes - 1) * step + lane + 1 : transfer->row_bytes;
   if (span > remaining ||
       (transfer->rows > 1 && row > (remaining - span) / (transfer->rows - 1)))
      return false;
   uint64_t rows = row * (transfer->rows - 1) + span;
   return transfer->slices == 1 || slice <= (remaining - rows) / (transfer->slices - 1);
}

VkResult
rd_execute_image_transfer(struct rd_device *device, const struct rd_image_transfer *transfer, bool clear)
{
   struct rd_memory *src = NULL, *dst = NULL;
   VkDeviceSize src_offset = 0, dst_offset = 0, src_size = 0, dst_size = 0;
   if (!rd_image_transfer_memory(transfer->dst_image, transfer->dst_buffer, &dst, &dst_offset, &dst_size) ||
       !rd_image_transfer_bounds(transfer->dst_offset, transfer->dst_row_pitch,
                                 transfer->dst_slice_pitch, transfer, dst_size,
                                 transfer->dst_step, transfer->dst_lane))
      return VK_ERROR_UNKNOWN;
   if (!clear && (!rd_image_transfer_memory(transfer->src_image, transfer->src_buffer, &src, &src_offset, &src_size) ||
       !rd_image_transfer_bounds(transfer->src_offset, transfer->src_row_pitch,
                                 transfer->src_slice_pitch, transfer, src_size,
                                 transfer->src_step, transfer->src_lane)))
      return VK_ERROR_UNKNOWN;
   if (clear && (!transfer->texel_size || transfer->texel_size > sizeof(transfer->clear) ||
                 transfer->row_bytes % transfer->texel_size))
      return VK_ERROR_UNKNOWN;
   if (transfer->copy_mask && (clear || transfer->row_bytes % 4 || transfer->lane_step))
      return VK_ERROR_UNKNOWN;
   if (transfer->lane_step && (clear || !transfer->src_step || !transfer->dst_step))
      return VK_ERROR_UNKNOWN;
   if (src && src->bo != dst->bo && fd_bo_cpu_prep(src->bo, device->pipe, FD_BO_PREP_READ | FD_BO_PREP_NOSYNC))
      return VK_ERROR_DEVICE_LOST;
   uint32_t flags = FD_BO_PREP_WRITE | FD_BO_PREP_NOSYNC;
   if ((src && src->bo == dst->bo) || transfer->copy_mask || transfer->lane_step)
      flags |= FD_BO_PREP_READ;
   if (fd_bo_cpu_prep(dst->bo, device->pipe, flags))
      return VK_ERROR_DEVICE_LOST;
   uint8_t *dst_map = fd_bo_map(dst->bo);
   const uint8_t *src_map = src ? fd_bo_map(src->bo) : NULL;
   if (!dst_map || (src && !src_map))
      return VK_ERROR_MEMORY_MAP_FAILED;
   __sync_synchronize();
   for (uint32_t z = 0; z < transfer->slices; z++) {
      for (uint32_t y = 0; y < transfer->rows; y++) {
         uint8_t *to = dst_map + dst_offset + transfer->dst_offset +
                       z * transfer->dst_slice_pitch + y * transfer->dst_row_pitch;
         if (clear) {
            for (VkDeviceSize x = 0; x < transfer->row_bytes; x += transfer->texel_size)
               memcpy(to + x, transfer->clear, transfer->texel_size);
         } else {
            const uint8_t *from = src_map + src_offset + transfer->src_offset +
                                  z * transfer->src_slice_pitch + y * transfer->src_row_pitch;
            if (transfer->lane_step) {
               for (VkDeviceSize x = 0; x < transfer->row_bytes; x++)
                  to[x * transfer->dst_step + transfer->dst_lane] =
                     from[x * transfer->src_step + transfer->src_lane];
            } else if (transfer->copy_mask) {
               for (VkDeviceSize x = 0; x < transfer->row_bytes; x += 4) {
                  uint32_t source, target;
                  memcpy(&source, from + x, 4);
                  memcpy(&target, to + x, 4);
                  target = (target & ~transfer->copy_mask) | (source & transfer->copy_mask);
                  memcpy(to + x, &target, 4);
               }
            } else {
               memmove(to, from, transfer->row_bytes);
            }
         }
      }
   }
   __sync_synchronize();
   return VK_SUCCESS;
}
