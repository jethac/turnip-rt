#include "rd_private.h"
#include "vk_descriptor_update_template.h"

static bool
rd_descriptor_type_supported(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      return true;
   default:
      return false;
   }
}

static bool
rd_descriptor_layout_supported(const VkDescriptorSetLayoutCreateInfo *info,
                                uint32_t *binding_count, uint32_t *descriptor_count)
{
   *binding_count = 0;
   *descriptor_count = 0;
   if (info->flags)
      return false;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
      if (!binding->descriptorCount)
         continue;
      if (!rd_descriptor_type_supported(binding->descriptorType) ||
          binding->descriptorCount > RD_MAX_BINDINGS - *descriptor_count)
         return false;
      *descriptor_count += binding->descriptorCount;
      (*binding_count)++;
   }
   return true;
}

VKAPI_ATTR void VKAPI_CALL
rd_GetDescriptorSetLayoutSupport(VkDevice handle,
   const VkDescriptorSetLayoutCreateInfo *info, VkDescriptorSetLayoutSupport *support)
{
   uint32_t bindings, descriptors;
   support->supported = rd_descriptor_layout_supported(info, &bindings, &descriptors);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateDescriptorSetLayout(VkDevice handle,
   const VkDescriptorSetLayoutCreateInfo *info,
   const VkAllocationCallbacks *allocator, VkDescriptorSetLayout *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   uint32_t bindings, descriptors;
   if (!rd_descriptor_layout_supported(info, &bindings, &descriptors))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   uint32_t immutable_count = 0;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
      if (binding->pImmutableSamplers &&
          (binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
           binding->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         immutable_count += binding->descriptorCount;
   }
   size_t size = sizeof(struct rd_set_layout) + bindings * sizeof(struct rd_binding) +
                 immutable_count * sizeof(struct vk_sampler_state);
   struct rd_set_layout *layout = vk_descriptor_set_layout_zalloc(&device->vk, size, info);
   if (!layout)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   layout->binding_count = bindings;
   layout->immutable_count = immutable_count;
   layout->immutable_sampler_states = (void *)&layout->bindings[bindings];
   layout->descriptor_count = descriptors;
   uint32_t slot = 0, index = 0;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      if (!info->pBindings[i].descriptorCount)
         continue;
      layout->bindings[index++] = (struct rd_binding) {
         .binding = info->pBindings[i].binding,
         .count = info->pBindings[i].descriptorCount,
         .slot = slot,
         .type = info->pBindings[i].descriptorType,
         .stages = info->pBindings[i].stageFlags,
         .immutable_offset = info->pBindings[i].pImmutableSamplers &&
            (info->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
             info->pBindings[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ? 0 : UINT32_MAX,
      };
      slot += info->pBindings[i].descriptorCount;
   }
   for (uint32_t i = 1; i < layout->binding_count; i++) {
      struct rd_binding binding = layout->bindings[i];
      uint32_t j = i;
      while (j && layout->bindings[j - 1].binding > binding.binding) {
         layout->bindings[j] = layout->bindings[j - 1];
         j--;
      }
      layout->bindings[j] = binding;
   }
   slot = 0;
   uint32_t immutable_offset = 0;
   for (uint32_t i = 0; i < layout->binding_count; i++) {
      if (layout->bindings[i].immutable_offset != UINT32_MAX) {
         layout->bindings[i].immutable_offset = immutable_offset;
         immutable_offset += layout->bindings[i].count;
      }
      layout->bindings[i].slot = slot;
      slot += layout->bindings[i].count;
      if (layout->bindings[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          layout->bindings[i].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         layout->vk.dynamic_descriptor_count += layout->bindings[i].count;
   }
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
      if (!binding->descriptorCount || !binding->pImmutableSamplers ||
          (binding->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER &&
           binding->descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         continue;
      for (uint32_t j = 0; j < layout->binding_count; j++) {
         if (layout->bindings[j].binding != binding->binding)
            continue;
         for (uint32_t k = 0; k < binding->descriptorCount; k++) {
            uint32_t offset = layout->bindings[j].immutable_offset + k;
            VK_FROM_HANDLE(rd_sampler, sampler, binding->pImmutableSamplers[k]);
            layout->immutable_sampler_states[offset] = sampler->state;
         }
      }
   }
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   _mesa_blake3_update(&hash, &layout->binding_count, sizeof(layout->binding_count));
   _mesa_blake3_update(&hash, layout->bindings,
                       layout->binding_count * sizeof(struct rd_binding));
   _mesa_blake3_update(&hash, &layout->immutable_count, sizeof(layout->immutable_count));
   _mesa_blake3_update(&hash, layout->immutable_sampler_states,
                       layout->immutable_count * sizeof(struct vk_sampler_state));
   _mesa_blake3_final(&hash, layout->vk.blake3);
   *out = rd_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

static const struct rd_binding *
rd_find_binding(struct rd_set_layout *layout, uint32_t binding)
{
   for (uint32_t i = 0; i < layout->binding_count; i++)
      if (layout->bindings[i].binding == binding)
         return &layout->bindings[i];
   return NULL;
}

static const struct vk_sampler_state *
rd_immutable_sampler(const struct rd_set_layout *layout, uint32_t slot)
{
   for (uint32_t i = 0; i < layout->binding_count; i++) {
      const struct rd_binding *binding = &layout->bindings[i];
      if (slot >= binding->slot && slot - binding->slot < binding->count) {
         if (binding->immutable_offset != UINT32_MAX)
            return &layout->immutable_sampler_states[binding->immutable_offset + slot - binding->slot];
         return NULL;
      }
   }
   return NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_CreateDescriptorPool(VkDevice handle, const VkDescriptorPoolCreateInfo *info,
                        const VkAllocationCallbacks *allocator, VkDescriptorPool *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   if (info->flags & ~VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   uint64_t count = 0;
   for (uint32_t i = 0; i < info->poolSizeCount; i++) {
      if (!rd_descriptor_type_supported(info->pPoolSizes[i].type))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      count += info->pPoolSizes[i].descriptorCount;
   }
   if (count > UINT32_MAX)
      return VK_ERROR_OUT_OF_POOL_MEMORY;
   struct rd_descriptor_pool *pool = vk_object_zalloc(&device->vk, allocator,
      sizeof(*pool), VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   pool->alloc = allocator ? *allocator : device->vk.alloc;
   pool->max_sets = info->maxSets;
   pool->max_descriptors = count;
   list_inithead(&pool->sets);
   *out = rd_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static void
rd_free_set(struct rd_device *device, struct rd_descriptor_set *set)
{
   list_del(&set->link);
   set->pool->used_sets--;
   set->pool->used_descriptors -= set->layout->descriptor_count;
   vk_descriptor_set_layout_unref(&device->vk, &set->layout->vk);
   vk_object_free(&device->vk, &set->pool->alloc, set);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_ResetDescriptorPool(VkDevice handle, VkDescriptorPool pool_handle,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_descriptor_pool, pool, pool_handle);
   list_for_each_entry_safe(struct rd_descriptor_set, set, &pool->sets, link)
      rd_free_set(device, set);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
rd_DestroyDescriptorPool(VkDevice handle, VkDescriptorPool pool_handle,
                         const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_descriptor_pool, pool, pool_handle);
   if (!pool)
      return;
   rd_ResetDescriptorPool(handle, pool_handle, 0);
   vk_object_free(&device->vk, allocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_AllocateDescriptorSets(VkDevice handle, const VkDescriptorSetAllocateInfo *info,
                          VkDescriptorSet *out)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   VK_FROM_HANDLE(rd_descriptor_pool, pool, info->descriptorPool);
   memset(out, 0, info->descriptorSetCount * sizeof(*out));
   uint64_t descriptors = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
      VK_FROM_HANDLE(rd_set_layout, layout, info->pSetLayouts[i]);
      descriptors += layout->descriptor_count;
   }
   if (info->descriptorSetCount > pool->max_sets - pool->used_sets ||
       descriptors > pool->max_descriptors - pool->used_descriptors)
      return VK_ERROR_OUT_OF_POOL_MEMORY;
   for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
      VK_FROM_HANDLE(rd_set_layout, layout, info->pSetLayouts[i]);
      struct rd_descriptor_set *set = vk_object_zalloc(&device->vk, &pool->alloc,
         sizeof(*set) + layout->descriptor_count * sizeof(struct rd_descriptor), VK_OBJECT_TYPE_DESCRIPTOR_SET);
      if (!set) {
         for (uint32_t j = 0; j < i; j++) {
            rd_free_set(device, rd_descriptor_set_from_handle(out[j]));
            out[j] = VK_NULL_HANDLE;
         }
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      vk_descriptor_set_layout_ref(&layout->vk);
      set->layout = layout;
      for (uint32_t j = 0; j < layout->binding_count; j++) {
         const struct rd_binding *binding = &layout->bindings[j];
         for (uint32_t k = 0; k < binding->count; k++) {
            uint32_t slot = binding->slot + k;
            set->descriptors[slot].type = binding->type;
            const struct vk_sampler_state *immutable = rd_immutable_sampler(layout, slot);
            if (immutable) {
               set->descriptors[slot].has_sampler = true;
               rd_sampler_descriptor(immutable, set->descriptors[slot].sampler);
            }
         }
      }
      set->pool = pool;
      pool->used_sets++;
      pool->used_descriptors += layout->descriptor_count;
      list_addtail(&set->link, &pool->sets);
      out[i] = rd_descriptor_set_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
rd_FreeDescriptorSets(VkDevice handle, VkDescriptorPool pool, uint32_t count,
                      const VkDescriptorSet *sets)
{
   VK_FROM_HANDLE(rd_device, device, handle);
   for (uint32_t i = 0; i < count; i++)
      if (sets[i])
         rd_free_set(device, rd_descriptor_set_from_handle(sets[i]));
   return VK_SUCCESS;
}

static void
rd_write_descriptor(struct rd_descriptor_set *set, uint32_t slot,
                     VkDescriptorType type, const void *data)
{
   const struct vk_sampler_state *immutable = rd_immutable_sampler(set->layout, slot);
   if (type == VK_DESCRIPTOR_TYPE_SAMPLER && immutable)
      return;
   struct rd_descriptor descriptor = { .type = type };
   switch (type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      VkDescriptorBufferInfo info;
      memcpy(&info, data, sizeof(info));
      descriptor.buffer = rd_buffer_from_handle(info.buffer);
      descriptor.offset = info.offset;
      descriptor.range = vk_buffer_range(&descriptor.buffer->vk, info.offset, info.range);
      break;
   }
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
      VkBufferView view;
      memcpy(&view, data, sizeof(view));
      descriptor.buffer_view = rd_buffer_view_from_handle(view);
      break;
   }
   default: {
      VkDescriptorImageInfo info;
      memcpy(&info, data, sizeof(info));
      if (type != VK_DESCRIPTOR_TYPE_SAMPLER)
         descriptor.image_view = rd_image_view_from_handle(info.imageView);
      if (type == VK_DESCRIPTOR_TYPE_SAMPLER ||
          type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
         if (immutable) {
            descriptor.has_sampler = true;
            rd_sampler_descriptor(immutable, descriptor.sampler);
         } else {
            VK_FROM_HANDLE(rd_sampler, sampler, info.sampler);
            descriptor.has_sampler = sampler != NULL;
            if (sampler)
               rd_sampler_descriptor(&sampler->state, descriptor.sampler);
         }
      }
      break;
   }
   }
   set->descriptors[slot] = descriptor;
}

VKAPI_ATTR void VKAPI_CALL
rd_UpdateDescriptorSets(VkDevice handle, uint32_t write_count,
                         const VkWriteDescriptorSet *writes,
                         uint32_t copy_count, const VkCopyDescriptorSet *copies)
{
   for (uint32_t i = 0; i < write_count; i++) {
      VK_FROM_HANDLE(rd_descriptor_set, set, writes[i].dstSet);
      const struct rd_binding *binding = rd_find_binding(set->layout, writes[i].dstBinding);
      assert(binding && writes[i].descriptorType == binding->type);
      assert(writes[i].dstArrayElement < binding->count);
      assert(writes[i].descriptorCount <= set->layout->descriptor_count -
             binding->slot - writes[i].dstArrayElement);
      for (uint32_t j = 0; j < writes[i].descriptorCount; j++) {
         const void *data;
         if (binding->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
             binding->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
             binding->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
             binding->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
            data = &writes[i].pBufferInfo[j];
         else if (binding->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                  binding->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
            data = &writes[i].pTexelBufferView[j];
         else
            data = writes[i].pImageInfo ? &writes[i].pImageInfo[j] : NULL;
         rd_write_descriptor(set, binding->slot + writes[i].dstArrayElement + j,
                              binding->type, data);
      }
   }
   for (uint32_t i = 0; i < copy_count; i++) {
      VK_FROM_HANDLE(rd_descriptor_set, src, copies[i].srcSet);
      VK_FROM_HANDLE(rd_descriptor_set, dst, copies[i].dstSet);
      const struct rd_binding *sb = rd_find_binding(src->layout, copies[i].srcBinding);
      const struct rd_binding *db = rd_find_binding(dst->layout, copies[i].dstBinding);
      assert(sb && db && sb->type == db->type);
      assert(copies[i].srcArrayElement < sb->count);
      assert(copies[i].descriptorCount <= src->layout->descriptor_count -
             sb->slot - copies[i].srcArrayElement);
      assert(copies[i].dstArrayElement < db->count);
      assert(copies[i].descriptorCount <= dst->layout->descriptor_count -
             db->slot - copies[i].dstArrayElement);
      uint32_t first = db->slot + copies[i].dstArrayElement;
      memmove(&dst->descriptors[first],
              &src->descriptors[sb->slot + copies[i].srcArrayElement],
              copies[i].descriptorCount * sizeof(struct rd_descriptor));
      for (uint32_t j = 0; j < copies[i].descriptorCount; j++) {
         uint32_t slot = first + j;
         const struct vk_sampler_state *immutable = rd_immutable_sampler(dst->layout, slot);
         if (immutable) {
            dst->descriptors[slot].has_sampler = true;
            rd_sampler_descriptor(immutable, dst->descriptors[slot].sampler);
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
rd_UpdateDescriptorSetWithTemplate(VkDevice handle, VkDescriptorSet set_handle,
   VkDescriptorUpdateTemplate template_handle, const void *data)
{
   VK_FROM_HANDLE(rd_descriptor_set, set, set_handle);
   VK_FROM_HANDLE(vk_descriptor_update_template, update, template_handle);
   for (uint32_t i = 0; i < update->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &update->entries[i];
      const struct rd_binding *binding = rd_find_binding(set->layout, entry->binding);
      assert(binding && entry->type == binding->type);
      assert(entry->array_element < binding->count);
      assert(entry->array_count <= set->layout->descriptor_count -
             binding->slot - entry->array_element);
      for (uint32_t j = 0; j < entry->array_count; j++)
         rd_write_descriptor(set, binding->slot + entry->array_element + j,
            entry->type, (const uint8_t *)data + entry->offset + j * entry->stride);
   }
}
