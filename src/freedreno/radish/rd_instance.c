#include "rd_private.h"

#include "util/os_misc.h"
#include "util/build_id.h"
#include "freedreno_uuid.h"
#include "util/u_memory.h"
#include "vk_util.h"
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice handle,
   const VkPhysicalDeviceExternalBufferInfo *info,
   VkExternalBufferProperties *properties)
{
   properties->externalMemoryProperties = (VkExternalMemoryProperties) {
      .externalMemoryFeatures = 0,
      .exportFromImportedHandleTypes = 0,
      .compatibleHandleTypes = info->handleType,
   };
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceExternalFenceProperties(VkPhysicalDevice handle,
   const VkPhysicalDeviceExternalFenceInfo *info,
   VkExternalFenceProperties *properties)
{
   properties->exportFromImportedHandleTypes = 0;
   properties->compatibleHandleTypes = 0;
   properties->externalFenceFeatures = 0;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceExternalSemaphoreProperties(VkPhysicalDevice handle,
   const VkPhysicalDeviceExternalSemaphoreInfo *info,
   VkExternalSemaphoreProperties *properties)
{
   properties->exportFromImportedHandleTypes = 0;
   properties->compatibleHandleTypes = 0;
   properties->externalSemaphoreFeatures = 0;
}

static const struct vk_instance_extension_table rd_instance_extensions = {
   .KHR_get_physical_device_properties2 = true,
   .KHR_device_group_creation = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
};

static const struct vk_device_extension_table rd_device_extensions = {
   .KHR_device_group = true,
   .KHR_descriptor_update_template = true,
   .KHR_maintenance3 = true,
   .KHR_bind_memory2 = true,
   .KHR_get_memory_requirements2 = true,
   .KHR_storage_buffer_storage_class = true,
   .KHR_16bit_storage = true,
   .KHR_8bit_storage = true,
   .KHR_shader_float16_int8 = true,
   .KHR_relaxed_block_layout = true,
   .KHR_sampler_mirror_clamp_to_edge = true,
};

VKAPI_ATTR VkResult VKAPI_CALL
rd_EnumerateInstanceVersion(uint32_t *version)
{
   *version = VK_API_VERSION_1_1;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_EnumerateInstanceExtensionProperties(const char *layer, uint32_t *count,
                                      VkExtensionProperties *properties)
{
   if (layer)
      return VK_ERROR_LAYER_NOT_PRESENT;
   return vk_enumerate_instance_extension_properties(&rd_instance_extensions,
                                                     count, properties);
}

static void
rd_destroy_physical(struct vk_physical_device *base)
{
   struct rd_physical_device *physical = container_of(base, struct rd_physical_device, vk);
   fd_device_del(physical->fd);
   close(physical->render_fd);
   vk_physical_device_finish(base);
   vk_free(&base->instance->alloc, physical);
}

static VkResult
rd_try_create_physical(struct vk_instance *instance, drmDevicePtr drm,
                       struct vk_physical_device **out)
{
   if (!(drm->available_nodes & (1 << DRM_NODE_RENDER)))
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   int render_fd = open(drm->nodes[DRM_NODE_RENDER], O_RDWR | O_CLOEXEC);
   if (render_fd < 0)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   drmVersionPtr version = drmGetVersion(render_fd);
   bool is_msm = version && !strcmp(version->name, "msm");
   drmFreeVersion(version);
   if (!is_msm) {
      close(render_fd);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   struct fd_device *fd = fd_device_new_dup(render_fd);
   if (!fd) {
      close(render_fd);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   struct fd_pipe *pipe = fd_pipe_new(fd, FD_PIPE_3D);
   if (!pipe) {
      fd_device_del(fd);
      close(render_fd);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   uint64_t gpu_id = 0;
   int ret = fd_pipe_get_param(pipe, FD_GPU_ID, &gpu_id);
   const struct fd_dev_id dev_id = *fd_pipe_dev_id(pipe);
   fd_pipe_del(pipe);
   if (ret || gpu_id != 506) {
      fd_device_del(fd);
      close(render_fd);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   struct rd_physical_device *physical = vk_zalloc(&instance->alloc,
      sizeof(*physical), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!physical) {
      fd_device_del(fd);
      close(render_fd);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   struct vk_physical_device_dispatch_table dispatch;
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch,
      &rd_physical_device_entrypoints, true);
   struct vk_properties properties = {
      .apiVersion = VK_API_VERSION_1_1,
      .driverVersion = 1,
      .vendorID = 0x5143,
      .deviceID = 506,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
      .subgroupSize = 16,
      .subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT,
      .maxMemoryAllocationCount = 4096,
      .bufferImageGranularity = 64,
      .discreteQueuePriorities = 2,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .maxPerSetDescriptors = 1024,
      .maxMemoryAllocationSize = UINT32_MAX - 4095ull,
      .maxImageDimension1D = 4096,
      .maxImageDimension2D = 4096,
      .maxImageDimension3D = 256,
      .maxImageDimensionCube = 4096,
      .maxImageArrayLayers = 256,
      .minTexelBufferOffsetAlignment = 64,
      .maxTexelBufferElements = 1 << 27,
      .maxSamplerAllocationCount = 4096,
      .maxSamplerLodBias = 16.0f,
      .maxSamplerAnisotropy = 1.0f,
      .pointSizeRange = {1.0f, 1.0f},
      .lineWidthRange = {1.0f, 1.0f},
      .maxViewports = 1,
      .maxViewportDimensions = {4096, 4096},
      .viewportBoundsRange = {-8192.0f, 8191.0f},
      .maxFramebufferWidth = 4096,
      .maxFramebufferHeight = 4096,
      .maxFramebufferLayers = 256,
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .maxColorAttachments = 4,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = 2047,
      .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 64,
      .maxFragmentInputComponents = 64,
      .maxFragmentOutputAttachments = 4,
      .maxFragmentCombinedOutputResources = 4,
      .maxDrawIndirectCount = 1,
      .maxDrawIndexedIndexValue = (1u << 24) - 1,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .subPixelPrecisionBits = 4,
      .subTexelPrecisionBits = 4,
      .mipmapPrecisionBits = 8,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      .maxBoundDescriptorSets = 4,
      .maxPerStageDescriptorSamplers = 16,
      .maxPerStageDescriptorUniformBuffers = 12,
      .maxPerStageDescriptorStorageBuffers = 5,
      .maxPerStageDescriptorSampledImages = 16,
      .maxPerStageDescriptorStorageImages = 4,
      .maxPerStageDescriptorInputAttachments = 4,
      .maxPerStageResources = 128,
      .maxDescriptorSetSamplers = 96,
      .maxDescriptorSetUniformBuffers = 72,
      .maxDescriptorSetUniformBuffersDynamic = 8,
      .maxDescriptorSetStorageBuffers = 24,
      .maxDescriptorSetStorageBuffersDynamic = 4,
      .maxDescriptorSetSampledImages = 96,
      .maxDescriptorSetStorageImages = 24,
      .maxDescriptorSetInputAttachments = 4,
      .timestampComputeAndGraphics = true,
      .timestampPeriod = 1000000000.0f / 19200000.0f,
      .maxStorageBufferRange = 128 * 1024 * 1024,
      .maxUniformBufferRange = 16384,
      .maxPushConstantsSize = 128,
      .minMemoryMapAlignment = 64,
      .minStorageBufferOffsetAlignment = 16,
      .minUniformBufferOffsetAlignment = 16,
      .nonCoherentAtomSize = 64,
      .maxComputeSharedMemorySize = RD_MAX_SHARED_SIZE,
      .maxComputeWorkGroupCount = {65535, 65535, 65535},
      .maxComputeWorkGroupInvocations = RD_MAX_WORKGROUP_INVOCATIONS,
      .maxComputeWorkGroupSize = {RD_MAX_WORKGROUP_INVOCATIONS, RD_MAX_WORKGROUP_INVOCATIONS, 64},
   };
   fd_get_device_uuid(properties.deviceUUID, &dev_id);
   fd_get_driver_uuid(properties.driverUUID);
   const struct build_id_note *build_id = build_id_find_nhdr_for_addr(rd_try_create_physical);
   if (build_id && build_id_length(build_id) >= VK_UUID_SIZE)
      memcpy(properties.driverUUID, build_id_data(build_id), VK_UUID_SIZE);
   memcpy(properties.pipelineCacheUUID, properties.driverUUID, VK_UUID_SIZE);
   properties.deviceNodeMask = 1;
   snprintf(properties.deviceName, sizeof(properties.deviceName),
            "Radish Adreno 506 (experimental compute)");
   physical->sync_type = vk_drm_syncobj_get_type(render_fd);
   const uint32_t timeline_sync_features = VK_SYNC_FEATURE_TIMELINE |
      VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
      VK_SYNC_FEATURE_WAIT_ANY | VK_SYNC_FEATURE_GPU_WAIT;
   const bool timeline =
      (physical->sync_type.features & timeline_sync_features) == timeline_sync_features &&
      physical->sync_type.get_value && physical->sync_type.signal &&
      physical->sync_type.wait_many;
   struct vk_device_extension_table extensions = rd_device_extensions;
   extensions.KHR_timeline_semaphore = timeline;
   properties.maxTimelineSemaphoreValueDifference = timeline ? UINT64_MAX : 0;
   const struct vk_features features = { .pipelineStatisticsQuery = true, .robustBufferAccess = true,
      .textureCompressionETC2 = true,
      .storageBuffer16BitAccess = true, .storageBuffer8BitAccess = true, .shaderInt8 = true, .timelineSemaphore = timeline };
   VkResult result = vk_physical_device_init(&physical->vk, instance, &extensions,
                                             &features, &properties, &dispatch);
   if (result != VK_SUCCESS) {
      vk_free(&instance->alloc, physical);
      fd_device_del(fd);
      close(render_fd);
      return result;
   }
   physical->fd = fd;
   physical->render_fd = render_fd;
   physical->sync_wait_many = physical->sync_type.wait_many;
   physical->sync_type.wait_many = rd_sync_wait_many;
   physical->sync_types[0] = &physical->sync_type;
   physical->vk.supported_sync_types = physical->sync_types;
   *out = &physical->vk;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateInstance(const VkInstanceCreateInfo *info,
                  const VkAllocationCallbacks *allocator, VkInstance *out)
{
   if (!allocator)
      allocator = vk_default_allocator();
   struct rd_instance *instance = vk_zalloc(allocator, sizeof(*instance), 8,
                                           VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct vk_instance_dispatch_table dispatch;
   vk_instance_dispatch_table_from_entrypoints(&dispatch,
                                               &rd_instance_entrypoints, true);
   VkResult result = vk_instance_init(&instance->vk, &rd_instance_extensions,
                                      &dispatch, info, allocator);
   if (result != VK_SUCCESS) {
      vk_free(allocator, instance);
      return result;
   }
   instance->vk.physical_devices.try_create_for_drm = rd_try_create_physical;
   instance->vk.physical_devices.destroy = rd_destroy_physical;
   *out = rd_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyInstance(VkInstance handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_instance, instance, handle);
   if (!instance)
      return;
   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
rd_GetInstanceProcAddr(VkInstance handle, const char *name)
{
   VK_FROM_HANDLE(rd_instance, instance, handle);
   return vk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                    &rd_instance_entrypoints, name);
}

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *name)
{
   return rd_GetInstanceProcAddr(instance, name);
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice handle,
   uint32_t *count, VkQueueFamilyProperties2 *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, properties, count);
   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = (VkQueueFamilyProperties) {
         .queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                       (rd_graphics_enabled() ? VK_QUEUE_GRAPHICS_BIT : 0),
         .queueCount = 1,
         .timestampValidBits = 64,
         .minImageTransferGranularity = {1, 1, 1},
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice handle,
                                     VkPhysicalDeviceMemoryProperties2 *out)
{
   uint64_t total = 0;
   os_get_total_physical_memory(&total);
   out->memoryProperties = (VkPhysicalDeviceMemoryProperties) {
      .memoryTypeCount = 1,
      .memoryTypes = {{ .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, .heapIndex = 0 }},
      .memoryHeapCount = 1,
      .memoryHeaps = {{ .size = total / 2, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT }},
   };
}

VKAPI_ATTR void VKAPI_CALL
rd_GetPhysicalDeviceSparseImageFormatProperties2(VkPhysicalDevice handle,
   const VkPhysicalDeviceSparseImageFormatInfo2 *info, uint32_t *count,
   VkSparseImageFormatProperties2 *properties)
{
   *count = 0;
}
