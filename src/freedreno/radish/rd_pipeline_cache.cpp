/* SPDX-License-Identifier: MIT */
#include "rd_private.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "compiler/nir/nir_serialize.h"
#include "ir3/ir3_compiler.h"
#include "ir3/ir3_nir.h"
#include "util/blob.h"
#include "util/build_id.h"
#include <cstdlib>
#include <cstring>


static constexpr size_t meta_a = offsetof(rd_pipeline, descriptor_count);
static constexpr size_t meta_a_size = offsetof(rd_pipeline, sampling_nir) - meta_a;
static constexpr size_t meta_b = offsetof(rd_pipeline, sampling_mask);
static constexpr size_t meta_b_size = sizeof(rd_pipeline) - meta_b;
static constexpr uint32_t cache_version = 2;
static constexpr size_t cache_max_entry = 64 * 1024 * 1024;
struct rd_cache_object {
   vk_pipeline_cache_object base;
   uint8_t key[BLAKE3_OUT_LEN];
   uint32_t size;
   uint8_t data[];
};
static void digest(const void *data, size_t size, uint8_t *out)
{
   blake3_hasher ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, data, size);
   _mesa_blake3_final(&ctx, out);
}
static bool rd_cache_key(rd_device *device, rd_pipeline *pipeline,
                      const VkPipelineShaderStageCreateInfo *stage,
                      VkPipelineCreateFlags flags, uint8_t *out)
{
   const build_id_note *id = build_id_find_nhdr_for_addr((const void *)rd_cache_key);
   if (!id || !build_id_length(id))
      return false;
   blake3_hasher ctx;
   _mesa_blake3_init(&ctx);
   const uint32_t abi[] = {cache_version, sizeof(rd_pipeline), sizeof(ir3_shader_variant),
      sizeof(ir3_shader_options), sizeof(struct ir3_const_state), uint32_t(meta_a_size), uint32_t(meta_b_size)};
   _mesa_blake3_update(&ctx, abi, sizeof(abi));
   _mesa_blake3_update(&ctx, build_id_data(id), build_id_length(id));
   _mesa_blake3_update(&ctx, device->vk.physical->properties.deviceUUID, VK_UUID_SIZE);
   _mesa_blake3_update(&ctx, &flags, sizeof(flags));
   uint32_t robust = device->vk.enabled_features.robustBufferAccess;
   _mesa_blake3_update(&ctx, &robust, sizeof(robust));
   uint64_t debug = ir3_shader_debug_hash_key();
   _mesa_blake3_update(&ctx, &debug, sizeof(debug));
   uint8_t shader[BLAKE3_OUT_LEN];
   vk_pipeline_hash_shader_stage(flags, stage, nullptr, shader);
   _mesa_blake3_update(&ctx, shader, sizeof(shader));
   const vk_pipeline_layout *layout = pipeline->layout;
   _mesa_blake3_update(&ctx, &layout->set_count, sizeof(layout->set_count));
   _mesa_blake3_update(&ctx, &layout->create_flags, sizeof(layout->create_flags));
   for (uint32_t i = 0; i < layout->set_count; i++) {
      const rd_set_layout *set = container_of(layout->set_layouts[i], const rd_set_layout, vk);
      _mesa_blake3_update(&ctx, &set->binding_count, sizeof(set->binding_count));
      _mesa_blake3_update(&ctx, set->bindings, set->binding_count * sizeof(rd_binding));
      _mesa_blake3_update(&ctx, &layout->dynamic_descriptor_offset[i], sizeof(uint32_t));
      
   }
   _mesa_blake3_update(&ctx, &layout->push_range_count, sizeof(layout->push_range_count));
   _mesa_blake3_update(&ctx, layout->push_ranges, layout->push_range_count * sizeof(VkPushConstantRange));
   _mesa_blake3_final(&ctx, out);
   return true;
}

static bool serialize(vk_pipeline_cache_object *base, blob *out)
{
   rd_cache_object *object = container_of(base, rd_cache_object, base);
   return blob_write_uint32(out, object->size) && blob_write_bytes(out, object->data, object->size);
}
static void destroy(vk_device *, vk_pipeline_cache_object *base)
{
   vk_pipeline_cache_object_finish(base);
   free(base);
}
static vk_pipeline_cache_object *deserialize(vk_pipeline_cache *, const void *, size_t, blob_reader *);
static const vk_pipeline_cache_object_ops cache_ops = {serialize, deserialize, destroy};
static vk_pipeline_cache_object *deserialize(vk_pipeline_cache *cache,
    const void *key, size_t key_size, blob_reader *in)
{
   uint32_t size = blob_read_uint32(in);
   if (key_size != BLAKE3_OUT_LEN || in->overrun || size < BLAKE3_OUT_LEN ||
       size > cache_max_entry || size > size_t(in->end - in->current))
      return nullptr;
   const uint8_t *data = static_cast<const uint8_t *>(blob_read_bytes(in, size));
   uint8_t hash[BLAKE3_OUT_LEN];
   digest(data + BLAKE3_OUT_LEN, size - BLAKE3_OUT_LEN, hash);
   if (memcmp(hash, data, sizeof(hash)))
      return nullptr;
   rd_cache_object *object = static_cast<rd_cache_object *>(calloc(1, sizeof(*object) + size));
   if (!object)
      return nullptr;
   memcpy(object->key, key, key_size);
   memcpy(object->data, data, size);
   object->size = size;
   vk_pipeline_cache_object_init(cache->base.device, &object->base, &cache_ops, object->key, sizeof(object->key));
   return &object->base;
}



static bool variant_bounds(rd_device *device, const void *data, size_t size)
{
   blob_reader in;
   blob_reader_init(&in, data, size);
   blob_skip_bytes(&in, sizeof(ir3_shader_key));
   if (blob_read_uint32(&in) != MESA_SHADER_COMPUTE)
      return false;
   blob_read_uint8(&in); 
   ir3_shader_variant v = {};
   blob_copy_bytes(&in, VARIANT_CACHE_PTR(&v), VARIANT_CACHE_SIZE);
   if (in.overrun || !v.info.size || v.info.size > size_t(in.end - in.current))
      return false;
   blob_skip_bytes(&in, v.info.size);
   blob_skip_bytes(&in, sizeof(struct ir3_const_state));
   if (!device->compiler->info->props.load_shader_consts_via_preamble) {
      uint32_t count = blob_read_uint32(&in);
      if (in.overrun || count > size_t(in.end - in.current) / sizeof(uint32_t))
         return false;
      blob_skip_bytes(&in, size_t(count) * sizeof(uint32_t));
   }
   return !in.overrun && in.current == in.end;
}

extern "C" bool
rd_pipeline_cache_restore(rd_device *device, rd_pipeline *pipeline,
    VkPipelineCache handle, const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, handle);
   uint8_t key[BLAKE3_OUT_LEN];
   if (!cache || !rd_cache_key(device, pipeline, stage, flags, key))
      return false;
   bool hit;
   vk_pipeline_cache_object *base = vk_pipeline_cache_lookup_object(cache, key, sizeof(key), &cache_ops, &hit);
   if (!base)
      return false;
   rd_cache_object *object = container_of(base, rd_cache_object, base);
   blob_reader in;
   blob_reader_init(&in, object->data + BLAKE3_OUT_LEN, object->size - BLAKE3_OUT_LEN);
   rd_pipeline saved = {};
   blob_copy_bytes(&in, reinterpret_cast<uint8_t *>(&saved) + meta_a, meta_a_size);
   blob_copy_bytes(&in, reinterpret_cast<uint8_t *>(&saved) + meta_b, meta_b_size);
   ir3_shader_options options;
   blob_copy_bytes(&in, &options, sizeof(options));
   uint32_t native_size = blob_read_uint32(&in);
   uint32_t nir_size = blob_read_uint32(&in);
   bool valid = !in.overrun && saved.resource_count <= RD_MAX_ACTIVE_RESOURCES &&
      saved.descriptor_count == saved.resource_count && saved.buffer_count <= 32 &&
      saved.resource_usage.texture_count <= 16 && saved.resource_usage.sampler_count <= 16 &&
      saved.resource_usage.image_count <= 16 && saved.compile_flags == flags &&
      uint64_t(native_size) + nir_size == uint64_t(in.end - in.current) &&
      variant_bounds(device, in.current, native_size);
   const ir3_shader_variant *variant = nullptr;
   nir_shader *nir = nullptr;
   if (valid) {
      blob_reader native;
      blob_reader_init(&native, in.current, native_size);
      ir3_blob_read reader{&native, device->compiler};
      reader.variant(variant);
      valid = !native.overrun && native.current == native.end && variant;
      if (valid && nir_size) {
         blob_reader nir_blob;
         blob_reader_init(&nir_blob, in.current + native_size, nir_size);
         nir = nir_deserialize(nullptr, ir3_get_compiler_options(device->compiler), &nir_blob);
         valid = nir && !nir_blob.overrun && nir_blob.current == nir_blob.end;
      }
   }
   if (valid) {
      memcpy(reinterpret_cast<uint8_t *>(pipeline) + meta_a, reinterpret_cast<uint8_t *>(&saved) + meta_a, meta_a_size);
      memcpy(reinterpret_cast<uint8_t *>(pipeline) + meta_b, reinterpret_cast<uint8_t *>(&saved) + meta_b, meta_b_size);
      pipeline->variant = const_cast<ir3_shader_variant *>(variant);
      pipeline->variant->shader_options = options;
      pipeline->sampling_nir = nir;
   } else {
      ralloc_free(const_cast<ir3_shader_variant *>(variant));
      ralloc_free(nir);
   }
   vk_pipeline_cache_object_unref(&device->vk, base);
   return valid;
}

extern "C" void
rd_pipeline_cache_store(rd_device *device, rd_pipeline *pipeline,
    VkPipelineCache handle, const VkPipelineShaderStageCreateInfo *stage, VkPipelineCreateFlags flags)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, handle);
   uint8_t key[BLAKE3_OUT_LEN];
   if (!cache || !rd_cache_key(device, pipeline, stage, flags, key))
      return;
   blob native, nir, payload;
   blob_init(&native); blob_init(&nir); blob_init(&payload);
   ir3_blob_write writer{&native};
   const ir3_shader_variant *variant = pipeline->variant;
   writer.variant(variant);
   if (pipeline->sampling_nir)
      nir_serialize(&nir, pipeline->sampling_nir, false);
   uint8_t hash[BLAKE3_OUT_LEN] = {};
   blob_write_bytes(&payload, hash, sizeof(hash));
   blob_write_bytes(&payload, reinterpret_cast<const uint8_t *>(pipeline) + meta_a, meta_a_size);
   blob_write_bytes(&payload, reinterpret_cast<const uint8_t *>(pipeline) + meta_b, meta_b_size);
   blob_write_bytes(&payload, &variant->shader_options, sizeof(variant->shader_options));
   blob_write_uint32(&payload, native.size); blob_write_uint32(&payload, nir.size);
   blob_write_bytes(&payload, native.data, native.size); blob_write_bytes(&payload, nir.data, nir.size);
   if (!native.out_of_memory && !nir.out_of_memory && !payload.out_of_memory && payload.size <= cache_max_entry) {
      digest(payload.data + sizeof(hash), payload.size - sizeof(hash), hash);
      memcpy(payload.data, hash, sizeof(hash));
      blob serialized;
      blob_init(&serialized);
      blob_write_uint32(&serialized, payload.size);
      blob_write_bytes(&serialized, payload.data, payload.size);
      if (!serialized.out_of_memory) {
         vk_pipeline_cache_object *object = vk_pipeline_cache_create_and_insert_object(cache,
            key, sizeof(key), serialized.data, serialized.size, &cache_ops);
         if (object)
            vk_pipeline_cache_object_unref(&device->vk, object);
      }
      blob_finish(&serialized);
   }
   blob_finish(&payload); blob_finish(&nir); blob_finish(&native);
}
