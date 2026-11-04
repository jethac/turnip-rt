#include "rd_private.h"

static bool
rd_transfer_range(struct rd_command_buffer *command, struct rd_buffer *buffer,
                   VkDeviceSize offset, VkDeviceSize size)
{
   if (!buffer || offset > buffer->vk.size || size > buffer->vk.size - offset) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return false;
   }
   return true;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdCopyBuffer(VkCommandBuffer handle, VkBuffer src_handle, VkBuffer dst_handle,
                  uint32_t count, const VkBufferCopy *regions)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_buffer, src, src_handle);
   VK_FROM_HANDLE(rd_buffer, dst, dst_handle);
   for (uint32_t i = 0; i < count; i++) {
      if (!rd_transfer_range(command, src, regions[i].srcOffset, regions[i].size) ||
          !rd_transfer_range(command, dst, regions[i].dstOffset, regions[i].size))
         return;
      struct rd_command *recorded = rd_command_append(command, RD_CMD_COPY);
      if (!recorded)
         return;
      recorded->transfer = (struct rd_transfer) {
         .src = src, .dst = dst, .src_offset = regions[i].srcOffset,
         .dst_offset = regions[i].dstOffset, .size = regions[i].size,
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdFillBuffer(VkCommandBuffer handle, VkBuffer dst_handle, VkDeviceSize offset,
                  VkDeviceSize size, uint32_t value)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_buffer, dst, dst_handle);
   if (size == VK_WHOLE_SIZE && offset <= dst->vk.size)
      size = (dst->vk.size - offset) & ~3ull;
   if ((offset & 3) || (size & 3)) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   if (!rd_transfer_range(command, dst, offset, size))
      return;
   struct rd_command *recorded = rd_command_append(command, RD_CMD_FILL);
   if (recorded)
      recorded->transfer = (struct rd_transfer) {
         .dst = dst, .dst_offset = offset, .size = size, .value = value,
      };
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdUpdateBuffer(VkCommandBuffer handle, VkBuffer dst_handle,
                    VkDeviceSize offset, VkDeviceSize size, const void *data)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_buffer, dst, dst_handle);
   if (size > 65536 || (offset & 3) || (size & 3) ||
       !rd_transfer_range(command, dst, offset, size)) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   struct rd_command *recorded = rd_command_append(command, RD_CMD_UPDATE);
   if (!recorded)
      return;
   void *copy = vk_alloc(&command->vk.pool->alloc, size, 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (!copy) {
      rd_set_error(&command->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   memcpy(copy, data, size);
   recorded->transfer = (struct rd_transfer) {
      .dst = dst, .dst_offset = offset, .size = size, .data = copy,
   };
}
