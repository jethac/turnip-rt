#include "rd_private.h"
#include <limits.h>
#include "util/u_math.h"

VKAPI_ATTR VkResult VKAPI_CALL
rd_AllocateMemory(VkDevice handle, const VkMemoryAllocateInfo *info,
                  const VkAllocationCallbacks *allocator, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   *out = VK_NULL_HANDLE;
   const VkMemoryAllocateFlagsInfo *flags =
      vk_find_struct_const(info->pNext, MEMORY_ALLOCATE_FLAGS_INFO);
   if (flags && ((flags->flags & ~VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT) ||
                 ((flags->flags & VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT) && flags->deviceMask != 1)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (info->memoryTypeIndex != 0 || !info->allocationSize ||
       info->allocationSize > UINT32_MAX - 4095)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct rd_memory *memory = vk_device_memory_create(&device->vk, info,
                                                      allocator, sizeof(*memory));
   if (!memory)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memory->bo = fd_bo_new(device->fd, align64(info->allocationSize, 4096),
                           FD_BO_HINT_BUFFER, "radish-memory");
   if (!memory->bo) {
      vk_device_memory_destroy(&device->vk, allocator, &memory->vk);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   memory->bo->bo_reuse = NO_CACHE;
   *out = rd_memory_to_handle(memory);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_FreeMemory(VkDevice handle, VkDeviceMemory memory_handle,
              const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_memory, memory, memory_handle);
   if (!memory)
      return;
   fd_bo_del(memory->bo);
   vk_device_memory_destroy(&device->vk, allocator, &memory->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_MapMemory2(VkDevice handle, const VkMemoryMapInfo *info, void **out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_memory, memory, info->memory);
   if (info->flags || memory->map || info->offset >= memory->vk.size ||
       (info->size != VK_WHOLE_SIZE && info->size > memory->vk.size - info->offset))
      return VK_ERROR_MEMORY_MAP_FAILED;
   if (fd_bo_cpu_prep(memory->bo, device->pipe,
                      FD_BO_PREP_READ | FD_BO_PREP_WRITE))
      return VK_ERROR_MEMORY_MAP_FAILED;
   void *base = fd_bo_map(memory->bo);
   if (!base)
      return VK_ERROR_MEMORY_MAP_FAILED;
   memory->map = base;
   *out = (uint8_t *)base + info->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_UnmapMemory2(VkDevice handle, const VkMemoryUnmapInfo *info)
{
   VK_FROM_HANDLE(rd_memory, memory, info->memory);
   __sync_synchronize();
   memory->map = NULL;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_FlushMappedMemoryRanges(VkDevice handle, uint32_t count,
                           const VkMappedMemoryRange *ranges)
{
   __sync_synchronize();
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_InvalidateMappedMemoryRanges(VkDevice handle, uint32_t count,
                                const VkMappedMemoryRange *ranges)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(rd_memory, memory, ranges[i].memory);
      if (fd_bo_cpu_prep(memory->bo, device->pipe, FD_BO_PREP_READ))
         return VK_ERROR_DEVICE_LOST;
   }
   __sync_synchronize();
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateBuffer(VkDevice handle, const VkBufferCreateInfo *info,
                const VkAllocationCallbacks *allocator, VkBuffer *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   if (info->flags || info->size > UINT32_MAX - 4095)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct rd_buffer *buffer = vk_buffer_create(&device->vk, info, allocator,
                                               sizeof(*buffer));
   if (!buffer)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *out = rd_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyBuffer(VkDevice handle, VkBuffer buffer_handle,
                 const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_buffer, buffer, buffer_handle);
   if (buffer)
      vk_buffer_destroy(&device->vk, allocator, &buffer->vk);
}

VKAPI_ATTR void VKAPI_CALL
rd_GetBufferMemoryRequirements2(VkDevice handle,
   const VkBufferMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(rd_buffer, buffer, info->buffer);
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(buffer->vk.size, 64), .alignment = 64, .memoryTypeBits = 1,
   };
   vk_foreach_struct(sType, next, out->pNext) {
      if (sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
         VkMemoryDedicatedRequirements *dedicated = (VkMemoryDedicatedRequirements *)next;
         dedicated->prefersDedicatedAllocation = false;
         dedicated->requiresDedicatedAllocation = false;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_BindBufferMemory2(VkDevice handle, uint32_t count,
                     const VkBindBufferMemoryInfo *infos)
{
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(rd_buffer, buffer, infos[i].buffer);
      VK_FROM_HANDLE(rd_memory, memory, infos[i].memory);
      const VkBindBufferMemoryDeviceGroupInfo *group =
         vk_find_struct_const(infos[i].pNext, BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO);
      if (group && (group->deviceIndexCount > 1 ||
          (group->deviceIndexCount && (!group->pDeviceIndices || group->pDeviceIndices[0]))))
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      if (!memory || infos[i].memoryOffset > memory->vk.size ||
          buffer->vk.size > memory->vk.size - infos[i].memoryOffset ||
          infos[i].memoryOffset % 64)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      buffer->memory = memory;
      buffer->offset = infos[i].memoryOffset;
      buffer->vk.device_address = fd_bo_get_iova(memory->bo) + buffer->offset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetDeviceMemoryCommitment(VkDevice handle, VkDeviceMemory memory_handle,
                             VkDeviceSize *committed)
{
   VK_FROM_HANDLE(rd_memory, memory, memory_handle);
   *committed = memory->vk.size;
}
