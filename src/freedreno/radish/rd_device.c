#include "rd_private.h"
#include "util/os_time.h"
#include <time.h>

void
rd_wake_device_waiters(struct vk_device *device)
{
   list_for_each_entry(struct vk_queue, queue, &device->queues, link) {
      mtx_lock(&queue->submit.mutex);
      cnd_broadcast(&queue->submit.pop);
      cnd_broadcast(&queue->submit.push);
      mtx_unlock(&queue->submit.mutex);
   }
}

VkResult
rd_sync_wait_many(struct vk_device *device, uint32_t count,
                   const struct vk_sync_wait *waits,
                   enum vk_sync_wait_flags flags, uint64_t deadline)
{
   struct rd_physical_device *physical =
      container_of(device->physical, struct rd_physical_device, vk);
   if (!physical->sync_wait_many)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const bool poll_pending = count && (flags & VK_SYNC_WAIT_PENDING) &&
      !(waits[0].sync->type->features & VK_SYNC_FEATURE_TIMELINE);
   for (;;) {
      if (vk_device_is_lost_no_report(device)) {
         rd_wake_device_waiters(device);
         return VK_ERROR_DEVICE_LOST;
      }
      uint64_t now = os_time_get_nano();
      uint64_t slice = now > UINT64_MAX - 100000000ull ? UINT64_MAX :
                       now + 100000000ull;
      if (slice > deadline)
         slice = deadline;
      VkResult result = physical->sync_wait_many(device, count, waits, flags,
                                                 poll_pending ? 0 : slice);
      if (vk_device_is_lost_no_report(device)) {
         rd_wake_device_waiters(device);
         return VK_ERROR_DEVICE_LOST;
      }
      if (result != VK_SUCCESS && result != VK_TIMEOUT) {
         vk_device_set_lost(device, "Radish sync wait failed");
         rd_wake_device_waiters(device);
         return VK_ERROR_DEVICE_LOST;
      }
      if (result != VK_TIMEOUT || (!poll_pending && slice == deadline))
         return result;
      if (poll_pending) {
         now = os_time_get_nano();
         if (now >= deadline)
            return VK_TIMEOUT;
         uint64_t delay = deadline - now;
         if (delay > 1000000ull)
            delay = 1000000ull;
         struct timespec sleep_time = { .tv_sec = 0, .tv_nsec = delay };
         nanosleep(&sleep_time, NULL);
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateDevice(VkPhysicalDevice handle, const VkDeviceCreateInfo *info,
                const VkAllocationCallbacks *allocator, VkDevice *out)
{
   VK_FROM_HANDLE(rd_physical_device, physical, handle);
   const VkDeviceGroupDeviceCreateInfo *group =
      vk_find_struct_const(info->pNext, DEVICE_GROUP_DEVICE_CREATE_INFO);
   if (group && group->physicalDeviceCount &&
       (group->physicalDeviceCount != 1 || group->pPhysicalDevices[0] != handle))
      return VK_ERROR_INITIALIZATION_FAILED;
   if (info->queueCreateInfoCount != 1 ||
       info->pQueueCreateInfos[0].queueFamilyIndex != 0 ||
       info->pQueueCreateInfos[0].queueCount != 1 ||
       info->pQueueCreateInfos[0].flags)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct rd_device *device = vk_zalloc2(&physical->vk.instance->alloc, allocator,
      sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct vk_device_dispatch_table dispatch;
   vk_device_dispatch_table_from_entrypoints(&dispatch, &rd_device_entrypoints, true);
   VkResult result = vk_device_init(&device->vk, &physical->vk, &dispatch,
                                    info, allocator);
   if (result != VK_SUCCESS) {
      vk_free2(&physical->vk.instance->alloc, allocator, device);
      return result;
   }
   device->queue_priority =
      info->pQueueCreateInfos[0].pQueuePriorities[0] >= 0.5f;
   device->fd = fd_device_ref(physical->fd);
   device->pipe = fd_pipe_new(device->fd, FD_PIPE_3D);
   if (!device->pipe) {
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto fail;
   }
   if (fd_pipe_get_param(device->pipe, FD_CTX_FAULTS, &device->faults)) {
      fd_pipe_del(device->pipe);
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto fail;
   }
   vk_device_set_drm_fd(&device->vk, physical->render_fd);
   vk_device_enable_threaded_submit(&device->vk);
   list_inithead(&device->active_queries);
   result = vk_queue_init(&device->queue, &device->vk,
                          &info->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS) {
      fd_pipe_del(device->pipe);
      goto fail;
   }
   device->queue.driver_submit = rd_queue_submit;
   device->vk.command_buffer_ops = &rd_command_buffer_ops;
   device->info = fd_dev_info(fd_pipe_dev_id(device->pipe));
   const struct ir3_compiler_options options = { .disable_cache = true };
   device->compiler = ir3_compiler_create(device->fd, fd_pipe_dev_id(device->pipe),
                                         &device->info, &options);
   device->control = fd_bo_new(device->fd, 4096, 0, "radish-control");
   if (!device->compiler || !device->control) {
      if (device->compiler) ir3_compiler_destroy(device->compiler);
      if (device->control) fd_bo_del(device->control);
      vk_queue_finish(&device->queue);
      fd_pipe_del(device->pipe);
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      goto fail;
   }
   device->compiler->nir_options.lower_device_index_to_zero = true;
   result = rd_init_border_colors(device);
   if (result == VK_SUCCESS)
      result = vk_queue_enable_submit_thread(&device->queue);
   if (result != VK_SUCCESS) {
      if (device->border_colors) fd_bo_del(device->border_colors);
      fd_bo_del(device->control);
      ir3_compiler_destroy(device->compiler);
      vk_queue_finish(&device->queue);
      fd_pipe_del(device->pipe);
      goto fail;
   }
   *out = rd_device_to_handle(device);
   return VK_SUCCESS;
fail:
   fd_device_del(device->fd);
   vk_device_finish(&device->vk);
   vk_free(&device->vk.alloc, device);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyDevice(VkDevice handle, const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   if (!device)
      return;
   vk_queue_finish(&device->queue);
   fd_bo_del(device->border_colors);
   fd_bo_del(device->control);
   ir3_compiler_destroy(device->compiler);
   fd_pipe_del(device->pipe);
   fd_device_del(device->fd);
   vk_device_finish(&device->vk);
   vk_free(&device->vk.alloc, device);
}
