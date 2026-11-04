#include "rd_private.h"
#include "util/u_math.h"
#include <stdint.h>
#include <string.h>
#include <time.h>

static bool
rd_query_range(const struct rd_query_pool *pool, uint32_t first, uint32_t count)
{
   return pool && first <= pool->vk.query_count &&
          count <= pool->vk.query_count - first;
}

static bool
rd_query_layout(const struct rd_query_pool *pool, uint32_t count,
                 VkDeviceSize stride, VkQueryResultFlags flags, size_t *bytes)
{
   const VkQueryResultFlags known = VK_QUERY_RESULT_64_BIT |
      VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
      VK_QUERY_RESULT_PARTIAL_BIT;
   if ((flags & ~known) ||
       (pool->vk.query_type == VK_QUERY_TYPE_TIMESTAMP &&
        (flags & VK_QUERY_RESULT_PARTIAL_BIT)))
      return false;
   const size_t width = flags & VK_QUERY_RESULT_64_BIT ? 8 : 4;
   const size_t words = pool->value_count +
      !!(flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   const size_t one = words * width;
   if ((stride % width) || (count > 1 && stride < one) ||
       (count > 1 && stride > (SIZE_MAX - one) / (count - 1)))
      return false;
   *bytes = count ? (size_t)(count - 1) * stride + one : 0;
   return true;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateQueryPool(VkDevice handle, const VkQueryPoolCreateInfo *info,
                   const VkAllocationCallbacks *allocator, VkQueryPool *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   *out = VK_NULL_HANDLE;
   if (info->flags || !info->queryCount ||
       (info->queryType != VK_QUERY_TYPE_OCCLUSION &&
        info->queryType != VK_QUERY_TYPE_TIMESTAMP &&
        info->queryType != VK_QUERY_TYPE_PIPELINE_STATISTICS))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (info->queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS &&
       (!device->vk.enabled_features.pipelineStatisticsQuery ||
        !info->pipelineStatistics || (info->pipelineStatistics & ~0x7ffu)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t slots_size, allocation_size;
   if (__builtin_mul_overflow((size_t)info->queryCount, sizeof(struct rd_query_slot),
                              &slots_size) ||
       __builtin_add_overflow(sizeof(struct rd_query_pool), slots_size,
                              &allocation_size))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct rd_query_pool *pool = vk_query_pool_create(&device->vk, info,
      allocator, allocation_size);
   if (!pool)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (mtx_init(&pool->mutex, mtx_plain) != thrd_success) {
      vk_query_pool_destroy(&device->vk, allocator, &pool->vk);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (cnd_init(&pool->condition) != thrd_success) {
      mtx_destroy(&pool->mutex);
      vk_query_pool_destroy(&device->vk, allocator, &pool->vk);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   pool->value_count = info->queryType == VK_QUERY_TYPE_PIPELINE_STATISTICS ?
      util_bitcount(info->pipelineStatistics) : 1;
   for (uint32_t i = 0; i < info->queryCount; i++) {
      pool->slots[i].pool = pool;
      list_inithead(&pool->slots[i].active_link);
   }
   if (info->queryType == VK_QUERY_TYPE_TIMESTAMP) {
      pool->bo = fd_bo_new(device->fd, 4096, 0, "radish-query");
      if (pool->bo)
         pool->map = fd_bo_map(pool->bo);
      if (!pool->map) {
         if (pool->bo)
            fd_bo_del(pool->bo);
         cnd_destroy(&pool->condition);
         mtx_destroy(&pool->mutex);
         vk_query_pool_destroy(&device->vk, allocator, &pool->vk);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
   }
   *out = rd_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyQueryPool(VkDevice handle, VkQueryPool pool_handle,
                    const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   if (!pool)
      return;
   for (uint32_t i = 0; i < pool->vk.query_count; i++)
      if (pool->slots[i].active)
         list_del(&pool->slots[i].active_link);
   if (pool->bo)
      fd_bo_del(pool->bo);
   cnd_destroy(&pool->condition);
   mtx_destroy(&pool->mutex);
   vk_query_pool_destroy(&device->vk, allocator, &pool->vk);
}

static VkResult
rd_query_wait(struct rd_device *device, struct rd_query_pool *pool,
               const struct rd_query_slot *slot)
{
   while (!slot->available) {
      if (vk_device_is_lost(&device->vk))
         return VK_ERROR_DEVICE_LOST;
      struct timespec deadline;
      if (!timespec_get(&deadline, TIME_UTC))
         return VK_ERROR_UNKNOWN;
      deadline.tv_nsec += 10000000;
      if (deadline.tv_nsec >= 1000000000) {
         deadline.tv_sec++;
         deadline.tv_nsec -= 1000000000;
      }
      int wait = cnd_timedwait(&pool->condition, &pool->mutex, &deadline);
      if (wait != thrd_success && wait != thrd_timedout)
         return VK_ERROR_UNKNOWN;
   }
   return VK_SUCCESS;
}

static void
rd_query_store(uint8_t *dst, uint64_t value, bool wide)
{
   if (wide)
      memcpy(dst, &value, sizeof(value));
   else {
      uint32_t low = (uint32_t)value;
      memcpy(dst, &low, sizeof(low));
   }
}

static VkResult
rd_query_results(struct rd_device *device, struct rd_query_pool *pool,
                  uint32_t first, uint32_t count, size_t size, void *data,
                  VkDeviceSize stride, VkQueryResultFlags flags)
{
   size_t required;
   if (!rd_query_range(pool, first, count) ||
       !rd_query_layout(pool, count, stride, flags, &required) ||
       required > size || (required && !data))
      return VK_ERROR_UNKNOWN;
   VkResult status = vk_device_check_status(&device->vk);
   if (status != VK_SUCCESS)
      return status;
   const bool wide = flags & VK_QUERY_RESULT_64_BIT;
   const size_t width = wide ? 8 : 4;
   mtx_lock(&pool->mutex);
   for (uint32_t i = 0; i < count; i++) {
      struct rd_query_slot *slot = &pool->slots[first + i];
      if (flags & VK_QUERY_RESULT_WAIT_BIT) {
         VkResult result = rd_query_wait(device, pool, slot);
         if (result != VK_SUCCESS) {
            mtx_unlock(&pool->mutex);
            return result;
         }
      }
      uint8_t *dst = (uint8_t *)data + (size_t)i * stride;
      if (slot->available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
         if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
            unsigned result = 0;
            for (unsigned bit = 0; bit < 11; bit++)
               if (pool->vk.pipeline_statistics & (1u << bit))
                  rd_query_store(dst + result++ * width, slot->values[bit], wide);
         } else {
            rd_query_store(dst, slot->values[0], wide);
         }
      } else {
         status = VK_NOT_READY;
      }
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         rd_query_store(dst + pool->value_count * width, slot->available, wide);
   }
   mtx_unlock(&pool->mutex);
   return status;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_GetQueryPoolResults(VkDevice handle, VkQueryPool pool_handle,
                       uint32_t first, uint32_t count, size_t size, void *data,
                       VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   return rd_query_results(device, pool, first, count, size, data, stride, flags);
}

static struct rd_query_command *
rd_record_query(struct rd_command_buffer *command, enum rd_command_type type,
                 struct rd_query_pool *pool, uint32_t first, uint32_t count)
{
   if (!rd_query_range(pool, first, count)) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return NULL;
   }
   struct rd_command *recorded = rd_command_append(command, type);
   if (!recorded)
      return NULL;
   recorded->query.pool = pool;
   recorded->query.first = first;
   recorded->query.count = count;
   return &recorded->query;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdResetQueryPool(VkCommandBuffer handle, VkQueryPool pool_handle,
                     uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   rd_record_query(command, RD_CMD_QUERY_RESET, pool, first, count);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdBeginQuery(VkCommandBuffer handle, VkQueryPool pool_handle,
                 uint32_t query, VkQueryControlFlags flags)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   if (pool && pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION) {
      if (!rd_query_range(pool, query, 1)) {
         rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
         return;
      }
      /* counts are exact whether or not PRECISE is asked for */
      command->occlusion_pool = pool;
      command->occlusion_query = query;
      if (!rd_render_query(command, pool, query, false))
         rd_record_query(command, RD_CMD_QUERY_BEGIN, pool, query, 1);
      return;
   }
   if (!pool || pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS || flags) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   rd_record_query(command, RD_CMD_QUERY_BEGIN, pool, query, 1);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdEndQuery(VkCommandBuffer handle, VkQueryPool pool_handle, uint32_t query)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   if (pool && pool->vk.query_type == VK_QUERY_TYPE_OCCLUSION) {
      command->occlusion_pool = NULL;
      if (!rd_render_query(command, pool, query, true))
         rd_record_query(command, RD_CMD_QUERY_END, pool, query, 1);
      return;
   }
   if (!pool || pool->vk.query_type != VK_QUERY_TYPE_PIPELINE_STATISTICS) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   rd_record_query(command, RD_CMD_QUERY_END, pool, query, 1);
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdWriteTimestamp2(VkCommandBuffer handle, VkPipelineStageFlags2 stage,
                      VkQueryPool pool_handle, uint32_t query)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   if (!pool || pool->vk.query_type != VK_QUERY_TYPE_TIMESTAMP) {
      rd_set_error(&command->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct rd_query_command *recorded =
      rd_record_query(command, RD_CMD_QUERY_TIMESTAMP, pool, query, 1);
   if (recorded)
      recorded->stage = stage;
}

VKAPI_ATTR void VKAPI_CALL
rd_CmdCopyQueryPoolResults(VkCommandBuffer handle, VkQueryPool pool_handle,
                           uint32_t first, uint32_t count, VkBuffer buffer_handle,
                           VkDeviceSize offset, VkDeviceSize stride,
                           VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(rd_command_buffer, command, handle);
   VK_FROM_HANDLE(rd_query_pool, pool, pool_handle);
   VK_FROM_HANDLE(rd_buffer, buffer, buffer_handle);
   size_t bytes;
   const size_t alignment = flags & VK_QUERY_RESULT_64_BIT ? 8 : 4;
   if (!pool || !buffer || !buffer->memory ||
       !(buffer->vk.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
       offset % alignment || !rd_query_layout(pool, count, stride, flags, &bytes) ||
       offset > buffer->vk.size || bytes > buffer->vk.size - offset) {
      rd_set_error(&command->vk, VK_ERROR_UNKNOWN);
      return;
   }
   struct rd_query_command *recorded =
      rd_record_query(command, RD_CMD_QUERY_COPY, pool, first, count);
   if (recorded) {
      recorded->dst = buffer;
      recorded->offset = offset;
      recorded->stride = stride;
      recorded->flags = flags;
   }
}

void
rd_query_account_dispatch(struct rd_device *device,
                           const struct rd_dispatch *dispatch)
{
   const struct ir3_shader_variant *v = dispatch->pipeline->variant;
   uint64_t invocations = (uint64_t)dispatch->grid[0] * dispatch->grid[1] *
      dispatch->grid[2] * v->local_size[0] * v->local_size[1] * v->local_size[2];
   list_for_each_entry(struct rd_query_slot, slot, &device->active_queries, active_link) {
      struct rd_query_pool *pool = slot->pool;
      mtx_lock(&pool->mutex);
      slot->values[10] += invocations;
      mtx_unlock(&pool->mutex);
   }
}

VkResult
rd_execute_query(struct rd_device *device, enum rd_command_type type,
                  const struct rd_query_command *command)
{
   struct rd_query_pool *pool = command->pool;
   if (!rd_query_range(pool, command->first, command->count))
      return VK_ERROR_UNKNOWN;
   if (type == RD_CMD_QUERY_COPY) {
      struct rd_buffer *dst = command->dst;
      size_t bytes;
      if (!rd_query_layout(pool, command->count, command->stride, command->flags, &bytes) ||
          !dst || !dst->memory || command->offset > dst->vk.size ||
          bytes > dst->vk.size - command->offset)
         return VK_ERROR_UNKNOWN;
      if (fd_bo_cpu_prep(dst->memory->bo, device->pipe, FD_BO_PREP_WRITE | FD_BO_PREP_NOSYNC))
         return VK_ERROR_DEVICE_LOST;
      uint8_t *map = fd_bo_map(dst->memory->bo);
      if (!map)
         return VK_ERROR_MEMORY_MAP_FAILED;
      VkResult result = rd_query_results(device, pool, command->first, command->count,
         bytes, map + dst->offset + command->offset, command->stride, command->flags);
      __sync_synchronize();
      return result == VK_NOT_READY ? VK_SUCCESS : result;
   }
   if (type == RD_CMD_QUERY_TIMESTAMP) {
      uint64_t value;
      VkResult result = rd_query_timestamp(device, pool, &value);
      if (result != VK_SUCCESS)
         return result;
      mtx_lock(&pool->mutex);
      pool->slots[command->first].values[0] = value;
      pool->slots[command->first].available = true;
      cnd_broadcast(&pool->condition);
      mtx_unlock(&pool->mutex);
      return VK_SUCCESS;
   }
   VkResult result = VK_SUCCESS;
   mtx_lock(&pool->mutex);
   for (uint32_t i = 0; i < command->count; i++) {
      struct rd_query_slot *slot = &pool->slots[command->first + i];
      if (type == RD_CMD_QUERY_END) {
         if (!slot->active) {
            result = VK_ERROR_UNKNOWN;
            break;
         }
         list_delinit(&slot->active_link);
         slot->active = false;
         slot->available = true;
         cnd_broadcast(&pool->condition);
      } else {
         if (slot->active) {
            result = VK_ERROR_UNKNOWN;
            break;
         }
         memset(slot->values, 0, sizeof(slot->values));
         slot->available = false;
         if (type == RD_CMD_QUERY_BEGIN) {
            slot->active = true;
            if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS)
               list_addtail(&slot->active_link, &device->active_queries);
         }
      }
   }
   mtx_unlock(&pool->mutex);
   return result;
}
