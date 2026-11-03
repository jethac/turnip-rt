#ifndef RD_NIR_RESOURCES_H
#define RD_NIR_RESOURCES_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct nir_shader;

struct rd_nir_resource_binding {
   uint32_t descriptor_set, binding, array_element, logical_slot, descriptor_slot;
   VkDescriptorType type;
   uint32_t buffer_slot, texture_slot, sampler_slot, image_slot;
};

struct rd_nir_resource_usage {
   uint64_t descriptors;
   uint32_t textures, samplers, images;
   uint32_t texture_count, sampler_count, image_count;
};

typedef const struct rd_nir_resource_binding *(*rd_nir_resource_resolver)(
   void *data, uint32_t descriptor_set, uint32_t binding, uint32_t element);

bool rd_nir_lower_compute_resources(struct nir_shader *shader,
   rd_nir_resource_resolver resolve, void *data,
   struct rd_nir_resource_usage *usage);

bool rd_nir_lower_robust_buffers(struct nir_shader *shader);
bool rd_nir_lower_subword_storage(struct nir_shader *shader);

#endif
