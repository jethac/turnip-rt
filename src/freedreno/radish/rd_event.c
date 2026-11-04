#include "rd_private.h"
#include <time.h>

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateEvent(VkDevice handle, const VkEventCreateInfo *info,
               const VkAllocationCallbacks *allocator, VkEvent *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   *out = VK_NULL_HANDLE;
   if (info->flags & ~VK_EVENT_CREATE_DEVICE_ONLY_BIT)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct rd_event *event = vk_object_zalloc(&device->vk, allocator,
      sizeof(*event), VK_OBJECT_TYPE_EVENT);
   if (!event)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (mtx_init(&event->mutex, mtx_plain) != thrd_success) {
      vk_object_free(&device->vk, allocator, event);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (cnd_init(&event->condition) != thrd_success) {
      mtx_destroy(&event->mutex);
      vk_object_free(&device->vk, allocator, event);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   *out = rd_event_to_handle(event);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyEvent(VkDevice handle, VkEvent event_handle,
                const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_event, event, event_handle);
   if (!event)
      return;
   cnd_destroy(&event->condition);
   mtx_destroy(&event->mutex);
   vk_object_free(&device->vk, allocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_GetEventStatus(VkDevice handle, VkEvent event_handle)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_event, event, event_handle);
   VkResult result = vk_device_check_status(&device->vk);
   if (result != VK_SUCCESS)
      return result;
   mtx_lock(&event->mutex);
   result = event->signaled ? VK_EVENT_SET : VK_EVENT_RESET;
   mtx_unlock(&event->mutex);
   return result;
}

static VkResult
rd_event_set_state(struct rd_device *device, struct rd_event *event, bool set)
{
   VkResult result = vk_device_check_status(&device->vk);
   if (result != VK_SUCCESS)
      return result;
   mtx_lock(&event->mutex);
   event->signaled = set;
   if (set)
      cnd_broadcast(&event->condition);
   mtx_unlock(&event->mutex);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_SetEvent(VkDevice handle, VkEvent event_handle)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_event, event, event_handle);
   return rd_event_set_state(device, event, true);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_ResetEvent(VkDevice handle, VkEvent event_handle)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_event, event, event_handle);
   return rd_event_set_state(device, event, false);
}

static void
rd_record_event(struct rd_command_buffer *command, enum rd_command_type type,
                 VkEvent event_handle)
{
   VK_FROM_HANDLE(rd_event, event, event_handle);
   if (!event) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   struct rd_command *recorded = rd_command_append(command, type);
   if (recorded)
      recorded->event.event = event;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdSetEvent2(VkCommandBuffer handle, VkEvent event,
                const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   rd_CmdPipelineBarrier2(handle, info);
   rd_record_event(command, RD_CMD_EVENT_SET, event);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdResetEvent2(VkCommandBuffer handle, VkEvent event,
                  VkPipelineStageFlags2 stage)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   rd_record_event(command, RD_CMD_EVENT_RESET, event);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdWaitEvents2(VkCommandBuffer handle, uint32_t count,
                  const VkEvent *events, const VkDependencyInfo *infos)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   for (uint32_t i = 0; i < count; i++)
      rd_record_event(command, RD_CMD_EVENT_WAIT, events[i]);
   for (uint32_t i = 0; i < count; i++)
      rd_CmdPipelineBarrier2(handle, &infos[i]);
}

VkResult
rd_execute_event(struct rd_device *device, enum rd_command_type type,
                  const struct rd_event_command *command)
{
   struct rd_event *event = command->event;
   if (type != RD_CMD_EVENT_WAIT)
      return rd_event_set_state(device, event, type == RD_CMD_EVENT_SET);
   VkResult result = VK_SUCCESS;
   mtx_lock(&event->mutex);
   while (!event->signaled) {
      if (vk_device_is_lost(&device->vk)) {
         result = VK_ERROR_DEVICE_LOST;
         break;
      }
      struct timespec deadline;
      if (!timespec_get(&deadline, TIME_UTC)) {
         result = VK_ERROR_UNKNOWN;
         break;
      }
      deadline.tv_nsec += 10000000;
      if (deadline.tv_nsec >= 1000000000) {
         deadline.tv_sec++;
         deadline.tv_nsec -= 1000000000;
      }
      int wait = cnd_timedwait(&event->condition, &event->mutex, &deadline);
      if (wait != thrd_success && wait != thrd_timedout) {
         result = VK_ERROR_UNKNOWN;
         break;
      }
   }
   mtx_unlock(&event->mutex);
   return result;
}
