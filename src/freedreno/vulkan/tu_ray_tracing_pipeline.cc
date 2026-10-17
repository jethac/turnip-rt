/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#include "tu_pipeline.h"

#include "compiler/spirv/spirv.h"
#include "nir/nir_builder.h"
#include "nir/nir_deref.h"
#include "util/hash_table.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "vk_pipeline.h"

#include "tu_device.h"
#include "tu_shader.h"
#include "ir3/ir3_nir.h"

static VkResult
tu_rt_compile_stages(struct tu_device *dev,
                     struct vk_pipeline_cache *cache,
                     struct tu_ray_tracing_pipeline *pipeline,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key);

static void
tu_rt_init_groups(struct tu_device *dev,
                  struct tu_ray_tracing_pipeline *pipeline,
                  const VkRayTracingPipelineCreateInfoKHR *create_info)
{
   uint32_t i = 0;
   for (; i < create_info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group_info =
         create_info->pGroups + i;
      struct tu_rt_group *dst = pipeline->groups + i;

      dst->recursive_index = VK_SHADER_UNUSED_KHR;
      dst->ahit_index = VK_SHADER_UNUSED_KHR;
      dst->isec_index = VK_SHADER_UNUSED_KHR;

      switch (group_info->type) {
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR:
         dst->recursive_index = group_info->generalShader;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR:
         dst->recursive_index = group_info->closestHitShader;
         dst->ahit_index = group_info->anyHitShader;
         break;
      case VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR:
         dst->recursive_index = group_info->closestHitShader;
         dst->isec_index = group_info->intersectionShader;
         if (dst->isec_index != VK_SHADER_UNUSED_KHR)
            dst->ahit_index = group_info->anyHitShader;
         break;
      default:
         UNREACHABLE("Unimplemented VkRayTracingShaderGroupTypeKHR");
      }

      uint32_t idx[3] = { dst->recursive_index, dst->ahit_index, dst->isec_index };
      struct mesa_blake3 h;
      unsigned char out[BLAKE3_KEY_LEN];
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, &group_info->type, sizeof(group_info->type));
      for (unsigned k = 0; k < 3; k++) {
         if (idx[k] == VK_SHADER_UNUSED_KHR) {
            uint32_t none = ~0u;
            _mesa_blake3_update(&h, &none, sizeof(none));
         } else {
            _mesa_blake3_update(&h, pipeline->stage_hashes + idx[k] * BLAKE3_KEY_LEN,
                                BLAKE3_KEY_LEN);
         }
      }
      _mesa_blake3_final(&h, out);
      memcpy(&dst->handle, out, sizeof(dst->handle));
      if (!dst->handle)
         dst->handle = 1;
   }

   if (!create_info->pLibraryInfo)
      return;

   uint32_t stage_base = create_info->stageCount;
   for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
      VK_FROM_HANDLE(tu_pipeline, lib_base,
                     create_info->pLibraryInfo->pLibraries[l]);
      struct tu_ray_tracing_pipeline *lib =
         tu_pipeline_to_ray_tracing(lib_base);

      for (uint32_t g = 0; g < lib->group_count; g++) {
         const struct tu_rt_group *src = lib->groups + g;
         struct tu_rt_group *dst = pipeline->groups + i;

         dst->handle = src->handle;
         dst->recursive_index = src->recursive_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->recursive_index;
         dst->ahit_index = src->ahit_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->ahit_index;
         dst->isec_index = src->isec_index == VK_SHADER_UNUSED_KHR ?
            VK_SHADER_UNUSED_KHR : stage_base + src->isec_index;
         i++;
      }
      stage_base += lib->stage_count;
   }
}

static bool
tu_rt_retag_constant_derefs(nir_shader *shader)
{
   bool progress = false;
   nir_foreach_function_impl (impl, shader) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->modes == nir_var_mem_constant) {
               deref->modes = nir_var_mem_global;
               progress = true;
            }
         }
      }
      nir_progress(progress, impl, nir_metadata_all);
   }
   return progress;
}

static bool
tu_rt_move_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   switch (instr->intrinsic) {
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_t_max:
      nir_instr_move(nir_before_impl(b->impl), &instr->instr);
      return true;
   default:
      return false;
   }
}

#define TU_RT_HIT_WORDS (TU_RT_HIT_ATTRIBS_SIZE / 4)
static const char tu_rt_hit_words_name[] = "tu_rt_hit_words";

static nir_variable *
tu_rt_hit_words_var(nir_shader *s)
{
   nir_foreach_variable_with_modes (var, s, nir_var_shader_temp) {
      if (var->name && !strcmp(var->name, tu_rt_hit_words_name))
         return var;
   }
   return nir_variable_create(s, nir_var_shader_temp,
                              glsl_array_type(glsl_uint_type(), TU_RT_HIT_WORDS, 4),
                              tu_rt_hit_words_name);
}

static nir_deref_instr *
tu_rt_hw(nir_builder *b, nir_variable *hw, nir_def *word)
{
   return nir_build_deref_array(b, nir_build_deref_var(b, hw), word);
}

static unsigned
tu_rt_elem_size(unsigned bit_size)
{
   return bit_size == 1 ? 4 : bit_size / 8;
}

static nir_def *
tu_rt_hw_load_comp(nir_builder *b, nir_variable *hw, nir_def *off,
                   unsigned c, unsigned bs)
{
   unsigned esz = tu_rt_elem_size(bs);
   nir_def *coff = nir_iadd_imm(b, off, c * esz);
   nir_def *word = nir_ushr_imm(b, coff, 2);
   switch (esz) {
   case 4: {
      nir_def *v = nir_load_deref(b, tu_rt_hw(b, hw, word));
      return bs == 1 ? nir_ine_imm(b, v, 0) : v;
   }
   case 8:
      return nir_pack_64_2x32_split(b, nir_load_deref(b, tu_rt_hw(b, hw, word)),
                                    nir_load_deref(b, tu_rt_hw(b, hw, nir_iadd_imm(b, word, 1))));
   default: {
      nir_def *w = nir_load_deref(b, tu_rt_hw(b, hw, word));
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, coff, 3), 8);
      return nir_u2uN(b, nir_ubitfield_extract(b, w, shift, nir_imm_int(b, bs)), bs);
   }
   }
}

static void
tu_rt_hw_store_comp(nir_builder *b, nir_variable *hw, nir_def *off,
                    unsigned c, nir_def *v)
{
   unsigned bs = v->bit_size;
   unsigned esz = tu_rt_elem_size(bs);
   nir_def *coff = nir_iadd_imm(b, off, c * esz);
   nir_def *word = nir_ushr_imm(b, coff, 2);
   switch (esz) {
   case 4:
      nir_store_deref(b, tu_rt_hw(b, hw, word), bs == 1 ? nir_b2i32(b, v) : v, 0x1);
      break;
   case 8:
      nir_store_deref(b, tu_rt_hw(b, hw, word), nir_unpack_64_2x32_split_x(b, v), 0x1);
      nir_store_deref(b, tu_rt_hw(b, hw, nir_iadd_imm(b, word, 1)),
                      nir_unpack_64_2x32_split_y(b, v), 0x1);
      break;
   default: {
      nir_deref_instr *d = tu_rt_hw(b, hw, word);
      nir_def *shift = nir_imul_imm(b, nir_iand_imm(b, coff, 3), 8);
      nir_store_deref(b, d, nir_bitfield_insert(b, nir_load_deref(b, d), nir_u2u32(b, v),
                                                shift, nir_imm_int(b, bs)), 0x1);
      break;
   }
   }
}

static bool
tu_rt_lower_hit_attribs(nir_shader *nir)
{
   bool has = false;
   nir_foreach_variable_with_modes (var, nir, nir_var_ray_hit_attrib)
      has = true;
   if (!has)
      return false;

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_ray_hit_attrib,
            glsl_get_natural_size_align_bytes);

   nir_variable *hw = tu_rt_hit_words_var(nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref &&
             intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_ray_hit_attrib))
            continue;

         b.cursor = nir_before_instr(instr);
         nir_def *off = nir_build_deref_offset(&b, deref, glsl_get_natural_size_align_bytes);

         if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_def *vals[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < intr->def.num_components; c++)
               vals[c] = tu_rt_hw_load_comp(&b, hw, off, c, intr->def.bit_size);
            nir_def_replace(&intr->def, nir_vec(&b, vals, intr->def.num_components));
         } else {
            nir_def *v = intr->src[1].ssa;
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < v->num_components; c++) {
               if (mask & (1u << c))
                  tu_rt_hw_store_comp(&b, hw, off, c, nir_channel(&b, v, c));
            }
            nir_instr_remove(instr);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_ray_hit_attrib, NULL);
   return true;
}

static VkResult
tu_rt_compile_stages(struct tu_device *dev,
                     struct vk_pipeline_cache *cache,
                     struct tu_ray_tracing_pipeline *pipeline,
                     const VkRayTracingPipelineCreateInfoKHR *create_info,
                     VkPipelineCreateFlags2KHR flags,
                     const struct tu_shader_key *key)
{
   const nir_shader_compiler_options *nir_options =
      ir3_get_compiler_options(dev->compiler);
   cache = cache ? cache : dev->mem_cache;
   bool can_cache = !(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR);

   uint32_t i = 0;
   for (; i < create_info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage_info =
         create_info->pStages + i;
      mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);

      static const char version[] = "tu-rt-stage-3";
      unsigned char nir_key[BLAKE3_KEY_LEN];
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, version, sizeof(version));
      _mesa_blake3_update(&h, pipeline->stage_hashes + i * BLAKE3_KEY_LEN, BLAKE3_KEY_LEN);
      _mesa_blake3_update(&h, key, sizeof(*key));
      enum ir3_shader_debug ir3_debug_key = ir3_shader_debug_hash_key();
      _mesa_blake3_update(&h, &ir3_debug_key, sizeof(ir3_debug_key));
      _mesa_blake3_final(&h, nir_key);

      nir_shader *nir = NULL;
      if (can_cache)
         nir = vk_pipeline_cache_lookup_nir(cache, nir_key, sizeof(nir_key),
                                            nir_options, NULL, pipeline->mem_ctx);
      if (!nir) {
         nir = tu_spirv_to_nir(dev, pipeline->mem_ctx, flags, stage_info, key, stage);
         if (!nir)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

         NIR_PASS(_, nir, tu_rt_retag_constant_derefs);
         NIR_PASS(_, nir, tu_rt_lower_hit_attribs);
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, tu_rt_move_intrinsic,
                  nir_metadata_control_flow, NULL);

         if (can_cache)
            vk_pipeline_cache_add_nir(cache, nir_key, sizeof(nir_key), nir);
      }

      pipeline->stages[i] = nir;
   }

   if (!create_info->pLibraryInfo)
      return VK_SUCCESS;

   for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
      VK_FROM_HANDLE(tu_pipeline, lib_base,
                     create_info->pLibraryInfo->pLibraries[l]);
      struct tu_ray_tracing_pipeline *lib =
         tu_pipeline_to_ray_tracing(lib_base);
      for (uint32_t s = 0; s < lib->stage_count; s++)
         pipeline->stages[i++] =
            nir_shader_clone(pipeline->mem_ctx, lib->stages[s]);
   }

   return VK_SUCCESS;
}

enum {
   TU_RT_USE_INSTANCE_ID = 1 << 0,
   TU_RT_USE_CUSTOM_INDEX = 1 << 1,
   TU_RT_USE_PRIMITIVE_ID = 1 << 2,
   TU_RT_USE_GEOMETRY_INDEX = 1 << 3,
   TU_RT_USE_OBJECT_ORIGIN = 1 << 4,
   TU_RT_USE_OBJECT_DIR = 1 << 5,
   TU_RT_USE_WTO = 1 << 6,
   TU_RT_USE_OTW = 1 << 7,
};

struct tu_rt_state {
   nir_variable *flags;
   nir_variable *cull_mask;
   nir_variable *sbt_offset;
   nir_variable *sbt_stride;
   nir_variable *origin;
   nir_variable *tmin;
   nir_variable *dir;
   nir_variable *tmax;

   nir_variable *instance_id;
   nir_variable *custom_index;
   nir_variable *primitive_id;
   nir_variable *geometry_index;
   nir_variable *hit_kind;
   nir_variable *object_origin;
   nir_variable *object_dir;
   nir_variable *wto[4];
   nir_variable *otw[4];

   nir_variable *shader_record_ptr;

   nir_variable *accept;
   nir_variable *terminate;
   nir_variable *opaque;
};

struct tu_rt_compiler {
   struct tu_ray_tracing_pipeline *pipeline;
   VkPipelineCreateFlags2KHR flags;
   struct tu_rt_state state;
   nir_variable *hit_words;

   struct hash_table *fn_ctx;
   nir_function **fn_cache;
   nir_function **isec_cache;

   uint32_t used;
   bool has_ahit;
   bool has_isec;
   unsigned max_call_depth;
};

static void
tu_rt_state_init(nir_shader *nir, struct tu_rt_state *s)
{
#define VAR(name, type) \
   s->name = nir_variable_create(nir, nir_var_shader_temp, type, #name)
   VAR(flags, glsl_uint_type());
   VAR(cull_mask, glsl_uint_type());
   VAR(sbt_offset, glsl_uint_type());
   VAR(sbt_stride, glsl_uint_type());
   VAR(origin, glsl_vec_type(3));
   VAR(tmin, glsl_float_type());
   VAR(dir, glsl_vec_type(3));
   VAR(tmax, glsl_float_type());
   VAR(instance_id, glsl_uint_type());
   VAR(custom_index, glsl_uint_type());
   VAR(primitive_id, glsl_uint_type());
   VAR(geometry_index, glsl_uint_type());
   VAR(hit_kind, glsl_uint_type());
   VAR(object_origin, glsl_vec_type(3));
   VAR(object_dir, glsl_vec_type(3));
   VAR(shader_record_ptr, glsl_uint64_t_type());
   VAR(accept, glsl_bool_type());
   VAR(terminate, glsl_bool_type());
   VAR(opaque, glsl_bool_type());
#undef VAR
   for (unsigned c = 0; c < 4; c++) {
      s->wto[c] = nir_variable_create(nir, nir_var_shader_temp,
                                      glsl_vec_type(3), "wto");
      s->otw[c] = nir_variable_create(nir, nir_var_shader_temp,
                                      glsl_vec_type(3), "otw");
   }
}

static uint32_t
tu_rt_scan_used(nir_shader *nir)
{
   uint32_t used = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_instance_id: used |= TU_RT_USE_INSTANCE_ID; break;
            case nir_intrinsic_load_ray_instance_custom_index: used |= TU_RT_USE_CUSTOM_INDEX; break;
            case nir_intrinsic_load_primitive_id: used |= TU_RT_USE_PRIMITIVE_ID; break;
            case nir_intrinsic_load_ray_geometry_index: used |= TU_RT_USE_GEOMETRY_INDEX; break;
            case nir_intrinsic_load_ray_object_origin: used |= TU_RT_USE_OBJECT_ORIGIN; break;
            case nir_intrinsic_load_ray_object_direction: used |= TU_RT_USE_OBJECT_DIR; break;
            case nir_intrinsic_load_ray_world_to_object: used |= TU_RT_USE_WTO; break;
            case nir_intrinsic_load_ray_object_to_world: used |= TU_RT_USE_OTW; break;
            default: break;
            }
         }
      }
   }
   return used;
}

static nir_def *
tu_rt_load_cmd(nir_builder *b, uint32_t offset, unsigned comps)
{
   nir_def *addr = nir_pack_64_2x32(b, nir_load_rt_cmd_addr_ir3(b));
   return nir_load_global(b, comps, 32, nir_iadd_imm(b, addr, offset),
                          .access = (gl_access_qualifier) (ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                          .align_mul = 4);
}

static nir_def *
tu_rt_load_cmd_addr(nir_builder *b, uint32_t offset)
{
   return nir_pack_64_2x32(b, tu_rt_load_cmd(b, offset, 2));
}

struct tu_rt_sbt_entry {
   nir_def *value;
   nir_def *shader_record_ptr;
};

static struct tu_rt_sbt_entry
tu_rt_load_sbt_entry(nir_builder *b, nir_def *index, uint32_t addr_offset)
{
   nir_def *addr = tu_rt_load_cmd_addr(b, addr_offset);

   if (index) {
      nir_def *stride =
         tu_rt_load_cmd(b, addr_offset + 2 * sizeof(VkDeviceSize), 1);
      addr = nir_iadd(b, addr, nir_u2u64(b, nir_imul(b, index, stride)));
   }

   return (struct tu_rt_sbt_entry) {
      .value = nir_load_global(b, 1, 32, addr,
                               .access = (gl_access_qualifier) (ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                               .align_mul = 4),
      .shader_record_ptr = nir_iadd_imm(b, addr, TU_RT_GROUP_HANDLE_SIZE),
   };
}

#define TU_RT_MAX_CALL_DEPTH 8

struct tu_rt_fn_ctx {
   mesa_shader_stage stage;
   unsigned ray_depth;
   unsigned call_depth;
};

static const struct tu_rt_fn_ctx tu_rt_main_ctx = { MESA_SHADER_COMPUTE, 0, 0 };

static const struct tu_rt_fn_ctx *
tu_rt_ctx(struct tu_rt_compiler *c, nir_function *function)
{
   struct hash_entry *e = _mesa_hash_table_search(c->fn_ctx, function);
   return e ? (const struct tu_rt_fn_ctx *) e->data : &tu_rt_main_ctx;
}

static void
tu_rt_set_ctx(struct tu_rt_compiler *c, nir_function *function,
              mesa_shader_stage stage, unsigned ray_depth, unsigned call_depth)
{
   struct tu_rt_fn_ctx *ctx = rzalloc(c->fn_ctx, struct tu_rt_fn_ctx);
   ctx->stage = stage;
   ctx->ray_depth = ray_depth;
   ctx->call_depth = call_depth;
   _mesa_hash_table_insert(c->fn_ctx, function, ctx);
}

static uint32_t
tu_rt_stage_index(struct tu_rt_compiler *c, nir_shader *stage)
{
   for (uint32_t i = 0; i < c->pipeline->stage_count; i++) {
      if (c->pipeline->stages[i] == stage)
         return i;
   }
   UNREACHABLE("unknown ray tracing stage");
}

static void
tu_rt_bind_call_data_param(nir_function_impl *impl)
{
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *param = nir_load_param(&b, 0);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (!nir_deref_mode_is(deref, nir_var_shader_call_data))
            continue;
         deref->modes = nir_var_function_temp;
         if (deref->deref_type == nir_deref_type_var) {
            b.cursor = nir_before_instr(&deref->instr);
            nir_deref_instr *cast =
               nir_build_deref_cast(&b, param, nir_var_function_temp,
                                    deref->var->type, 0);
            nir_def_replace(&deref->def, &cast->def);
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);
}

static nir_function *
tu_rt_clone_stage(struct tu_rt_compiler *c, nir_shader *dst, nir_shader *stage)
{
   nir_function_impl *copy =
      nir_function_impl_clone(dst, nir_shader_get_entrypoint(stage));

   struct hash_table *var_remap = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_block (block, copy) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             deref->var->data.mode == nir_var_function_temp ||
             deref->var->data.mode == nir_var_shader_call_data)
            continue;
         struct hash_entry *entry =
            _mesa_hash_table_search(var_remap, deref->var);
         if (!entry) {
            nir_variable *new_var;
            if (deref->var->name &&
                !strcmp(deref->var->name, tu_rt_hit_words_name)) {
               new_var = c->hit_words;
            } else {
               new_var = nir_variable_clone(deref->var, dst);
               nir_shader_add_variable(dst, new_var);
            }
            entry = _mesa_hash_table_insert(var_remap, deref->var, new_var);
         }
         deref->var = (nir_variable *) entry->data;
      }
   }
   ralloc_free(var_remap);

   nir_function *function =
      nir_function_create(dst, _mesa_shader_stage_to_string(stage->info.stage));
   function->num_params = 1;
   function->params = rzalloc_array(dst, nir_parameter, 1);
   function->params[0].num_components = 1;
   function->params[0].bit_size = 32;
   nir_function_set_impl(function, copy);
   tu_rt_bind_call_data_param(copy);
   return function;
}

static void
tu_rt_call_stage(nir_builder *b, struct tu_rt_compiler *c, nir_shader *stage,
                 nir_def *arg, unsigned ray_depth, unsigned call_depth)
{
   uint32_t slot = (tu_rt_stage_index(c, stage) * 2 + ray_depth) *
                   (TU_RT_MAX_CALL_DEPTH + 1) + call_depth;
   nir_function *function = c->fn_cache[slot];
   if (!function) {
      function = tu_rt_clone_stage(c, b->shader, stage);
      tu_rt_set_ctx(c, function, stage->info.stage, ray_depth, call_depth);
      c->fn_cache[slot] = function;
   }
   nir_build_call(b, function, 1, &arg);
}

static bool
tu_rt_handle_seen(struct tu_rt_compiler *c, uint32_t upto, uint32_t handle,
                  uint32_t mask_kind)
{
   for (uint32_t j = 0; j < upto; j++) {
      const struct tu_rt_group *g = c->pipeline->groups + j;
      if (g->handle != handle)
         continue;
      if (mask_kind == 0 && g->recursive_index != VK_SHADER_UNUSED_KHR)
         return true;
      if (mask_kind == 1 && g->ahit_index != VK_SHADER_UNUSED_KHR &&
          g->isec_index == VK_SHADER_UNUSED_KHR)
         return true;
      if (mask_kind == 2 && g->isec_index != VK_SHADER_UNUSED_KHR)
         return true;
   }
   return false;
}

static nir_shader *
tu_rt_group_stage(struct tu_rt_compiler *c, uint32_t index)
{
   return index == VK_SHADER_UNUSED_KHR ? NULL : c->pipeline->stages[index];
}

static void
tu_rt_dispatch_recursive(nir_builder *b, struct tu_rt_compiler *c,
                         nir_def *handle, mesa_shader_stage want, nir_def *arg,
                         unsigned ray_depth, unsigned call_depth)
{
   for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
      struct tu_rt_group *group = c->pipeline->groups + i;
      nir_shader *stage = tu_rt_group_stage(c, group->recursive_index);
      if (!stage || stage->info.stage != want ||
          tu_rt_handle_seen(c, i, group->handle, 0))
         continue;
      nir_push_if(b, nir_ieq_imm(b, handle, group->handle));
      tu_rt_call_stage(b, c, stage, arg, ray_depth, call_depth);
      nir_pop_if(b, NULL);
   }
}

static void
tu_rt_load_hit_state(nir_builder *b, struct tu_rt_compiler *c,
                     nir_def *rq, bool committed)
{
   struct tu_rt_state *s = &c->state;
   uint32_t u = c->used;

#define RQ(n, bits, val, ...) \
   nir_rq_load(b, n, bits, rq, .ray_query_value = nir_ray_query_value_##val, \
               .committed = committed, ##__VA_ARGS__)

   if (u & TU_RT_USE_INSTANCE_ID)
      nir_store_var(b, s->instance_id, RQ(1, 32, intersection_instance_id), 0x1);
   if (u & TU_RT_USE_CUSTOM_INDEX)
      nir_store_var(b, s->custom_index,
                    RQ(1, 32, intersection_instance_custom_index), 0x1);
   if (u & TU_RT_USE_PRIMITIVE_ID)
      nir_store_var(b, s->primitive_id,
                    RQ(1, 32, intersection_primitive_index), 0x1);
   if (u & TU_RT_USE_GEOMETRY_INDEX)
      nir_store_var(b, s->geometry_index,
                    RQ(1, 32, intersection_geometry_index), 0x1);
   if (u & TU_RT_USE_OBJECT_ORIGIN)
      nir_store_var(b, s->object_origin,
                    RQ(3, 32, intersection_object_ray_origin), 0x7);
   if (u & TU_RT_USE_OBJECT_DIR)
      nir_store_var(b, s->object_dir,
                    RQ(3, 32, intersection_object_ray_direction), 0x7);
   for (unsigned col = 0; col < 4; col++) {
      if (u & TU_RT_USE_WTO)
         nir_store_var(b, s->wto[col],
                       RQ(3, 32, intersection_world_to_object, .column = col), 0x7);
      if (u & TU_RT_USE_OTW)
         nir_store_var(b, s->otw[col],
                       RQ(3, 32, intersection_object_to_world, .column = col), 0x7);
   }
#undef RQ
}

static nir_def *
tu_rt_sbt_index(nir_builder *b, struct tu_rt_compiler *c, nir_def *rq,
                bool committed)
{
   struct tu_rt_state *s = &c->state;
   nir_def *instance_sbt =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_instance_sbt_index,
                  .committed = committed);
   nir_def *geometry =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_geometry_index,
                  .committed = committed);
   return nir_iadd(b, nir_iadd(b, nir_load_var(b, s->sbt_offset), instance_sbt),
                   nir_imul(b, nir_load_var(b, s->sbt_stride), geometry));
}

static nir_def *
tu_rt_triangle_hit_kind(nir_builder *b, nir_def *rq, bool committed)
{
   nir_def *front =
      nir_rq_load(b, 1, 1, rq,
                  .ray_query_value = nir_ray_query_value_intersection_front_face,
                  .committed = committed);
   return nir_bcsel(b, front, nir_imm_int(b, 0xFE), nir_imm_int(b, 0xFF));
}

static void
tu_rt_store_bary(nir_builder *b, struct tu_rt_compiler *c, nir_def *rq,
                 bool committed)
{
   nir_def *bary =
      nir_rq_load(b, 2, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_barycentrics,
                  .committed = committed);
   for (unsigned i = 0; i < 2; i++)
      nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)),
                      nir_channel(b, bary, i), 0x1);
}

static nir_function *tu_rt_get_isec_function(nir_builder *b,
                                             struct tu_rt_compiler *c,
                                             uint32_t group,
                                             unsigned call_depth);

static void
tu_rt_trace_ray(nir_builder *b, struct tu_rt_compiler *c,
                nir_intrinsic_instr *instr)
{
   struct tu_rt_state *s = &c->state;
   const uint32_t hit_sbt =
      offsetof(VkTraceRaysIndirectCommand2KHR, hitShaderBindingTableAddress);

   nir_def *accel_struct = instr->src[0].ssa;
   nir_def *flags = instr->src[1].ssa;
   nir_def *cull_mask = instr->src[2].ssa;
   nir_def *sbt_offset = nir_iand_imm(b, instr->src[3].ssa, 0xF);
   nir_def *sbt_stride = nir_iand_imm(b, instr->src[4].ssa, 0xF);
   nir_def *miss_index = nir_iand_imm(b, instr->src[5].ssa, 0xFFFF);
   nir_def *origin = instr->src[6].ssa;
   nir_def *tmin = instr->src[7].ssa;
   nir_def *dir = instr->src[8].ssa;
   nir_def *tmax = instr->src[9].ssa;
   nir_def *payload = instr->src[10].ssa;
   const unsigned cd = tu_rt_ctx(c, b->impl->function)->call_depth;

   nir_store_var(b, s->flags, flags, 0x1);
   nir_store_var(b, s->cull_mask, cull_mask, 0x1);
   nir_store_var(b, s->sbt_offset, sbt_offset, 0x1);
   nir_store_var(b, s->sbt_stride, sbt_stride, 0x1);
   nir_store_var(b, s->origin, origin, 0x7);
   nir_store_var(b, s->tmin, tmin, 0x1);
   nir_store_var(b, s->dir, dir, 0x7);
   nir_store_var(b, s->tmax, tmax, 0x1);

   nir_variable *rq_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "trace_rq");
   rq_var->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(b, rq_var)->def;

   nir_variable *committed_hit_kind =
      nir_local_variable_create(b->impl, glsl_uint_type(), "committed_hit_kind");
   nir_store_var(b, committed_hit_kind, nir_imm_int(b, 0), 0x1);

   nir_def *rq_flags =
      nir_iand_imm(b, flags, ~SpvRayFlagsSkipClosestHitShaderKHRMask);
   if (c->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR) {
      rq_flags = nir_ior_imm(b, nir_iand_imm(b, rq_flags,
                                             ~(SpvRayFlagsCullBackFacingTrianglesKHRMask |
                                               SpvRayFlagsCullFrontFacingTrianglesKHRMask)),
                             SpvRayFlagsSkipTrianglesKHRMask);
   }
   if (c->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
      rq_flags = nir_ior_imm(b, rq_flags, SpvRayFlagsSkipAABBsKHRMask);
   if (!c->has_isec) {
      rq_flags = nir_bcsel(b, nir_test_mask(b, rq_flags, SpvRayFlagsSkipTrianglesKHRMask),
                           rq_flags,
                           nir_ior_imm(b, rq_flags, SpvRayFlagsSkipAABBsKHRMask));
   }
   if (!c->has_ahit) {
      nir_def *forced =
         nir_ior_imm(b, nir_iand_imm(b, rq_flags, ~SpvRayFlagsNoOpaqueKHRMask),
                     SpvRayFlagsOpaqueKHRMask);
      nir_def *culls_opacity =
         nir_test_mask(b, flags, SpvRayFlagsCullOpaqueKHRMask |
                                 SpvRayFlagsCullNoOpaqueKHRMask);
      rq_flags = nir_bcsel(b, culls_opacity, rq_flags, forced);
   }

   nir_rq_initialize(b, rq, accel_struct, rq_flags, cull_mask, origin, tmin,
                     dir, tmax);

   nir_loop *loop = nir_push_loop(b);
   {
      nir_break_if(b, nir_inot(b, nir_rq_proceed(b, 1, rq)));

      nir_def *ctype =
         nir_rq_load(b, 1, 32, rq,
                     .ray_query_value = nir_ray_query_value_intersection_type,
                     .committed = false);

      nir_push_if(b, nir_ieq_imm(b, ctype, 0));
      if (!c->has_ahit) {
         nir_rq_confirm_intersection(b, rq);
      } else {
         tu_rt_load_hit_state(b, c, rq, false);
         struct tu_rt_sbt_entry entry =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, false), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, entry.shader_record_ptr, 0x1);

         nir_def *t = nir_rq_load(b, 1, 32, rq,
                                  .ray_query_value = nir_ray_query_value_intersection_t,
                                  .committed = false);
         nir_def *prev_tmax = nir_load_var(b, s->tmax);
         nir_def *prev_w0 = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 0)));
         nir_def *prev_w1 = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 1)));

         nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
         nir_store_var(b, s->terminate, nir_imm_false(b), 0x1);
         nir_store_var(b, s->tmax, t, 0x1);
         nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, false), 0x1);
         tu_rt_store_bary(b, c, rq, false);

         for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
            nir_shader *ahit = tu_rt_group_stage(c, c->pipeline->groups[i].ahit_index);
            if (!ahit || c->pipeline->groups[i].isec_index != VK_SHADER_UNUSED_KHR ||
                tu_rt_handle_seen(c, i, c->pipeline->groups[i].handle, 1))
               continue;
            nir_push_if(b, nir_ieq_imm(b, entry.value, c->pipeline->groups[i].handle));
            tu_rt_call_stage(b, c, ahit, payload, 1, cd);
            nir_pop_if(b, NULL);
         }

         nir_push_if(b, nir_load_var(b, s->accept));
         {
            nir_rq_confirm_intersection(b, rq);
            nir_push_if(b, nir_load_var(b, s->terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_push_else(b, NULL);
         {
            nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 0)), prev_w0, 0x1);
            nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, 1)), prev_w1, 0x1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
      }
      nir_push_else(b, NULL);
      if (c->has_isec) {
         tu_rt_load_hit_state(b, c, rq, false);
         struct tu_rt_sbt_entry entry =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, false), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, entry.shader_record_ptr, 0x1);

         nir_def *prev_words[TU_RT_HIT_WORDS];
         for (unsigned i = 0; i < TU_RT_HIT_WORDS; i++)
            prev_words[i] = nir_load_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)));

         nir_def *prev_tmax = nir_load_var(b, s->tmax);
         nir_store_var(b, s->accept, nir_imm_false(b), 0x1);
         nir_store_var(b, s->terminate,
                       nir_test_mask(b, flags, SpvRayFlagsTerminateOnFirstHitKHRMask), 0x1);
         nir_store_var(b, s->opaque,
                       nir_rq_load(b, 1, 1, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_candidate_aabb_opaque,
                                   .committed = false),
                       0x1);
         nir_store_var(b, s->tmax,
                       nir_rq_load(b, 1, 32, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_t,
                                   .committed = true),
                       0x1);

         for (uint32_t i = 0; i < c->pipeline->group_count; i++) {
            if (c->pipeline->groups[i].isec_index == VK_SHADER_UNUSED_KHR ||
                tu_rt_handle_seen(c, i, c->pipeline->groups[i].handle, 2))
               continue;
            nir_push_if(b, nir_ieq_imm(b, entry.value, c->pipeline->groups[i].handle));
            nir_build_call(b, tu_rt_get_isec_function(b, c, i, cd), 1, &payload);
            nir_pop_if(b, NULL);
         }

         nir_push_if(b, nir_load_var(b, s->accept));
         {
            nir_rq_generate_intersection(b, rq, nir_load_var(b, s->tmax));
            nir_store_var(b, committed_hit_kind, nir_load_var(b, s->hit_kind), 0x1);
            nir_push_if(b, nir_load_var(b, s->terminate));
            nir_rq_terminate(b, rq);
            nir_pop_if(b, NULL);
         }
         nir_push_else(b, NULL);
         {
            for (unsigned i = 0; i < TU_RT_HIT_WORDS; i++)
               nir_store_deref(b, tu_rt_hw(b, c->hit_words, nir_imm_int(b, i)),
                               prev_words[i], 0x1);
         }
         nir_pop_if(b, NULL);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, loop);

   nir_def *committed_type =
      nir_rq_load(b, 1, 32, rq,
                  .ray_query_value = nir_ray_query_value_intersection_type,
                  .committed = true);

   nir_push_if(b, nir_ine_imm(b, committed_type, 0));
   {
      nir_def *skip_chit =
         nir_test_mask(b, flags, SpvRayFlagsSkipClosestHitShaderKHRMask);
      nir_push_if(b, nir_inot(b, skip_chit));
      {
         tu_rt_load_hit_state(b, c, rq, true);
         nir_store_var(b, s->tmax,
                       nir_rq_load(b, 1, 32, rq,
                                   .ray_query_value = nir_ray_query_value_intersection_t,
                                   .committed = true),
                       0x1);

         if (c->has_isec) {
            nir_push_if(b, nir_ieq_imm(b, committed_type, 1));
            {
               nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
               tu_rt_store_bary(b, c, rq, true);
            }
            nir_push_else(b, NULL);
            {
               nir_store_var(b, s->hit_kind, nir_load_var(b, committed_hit_kind), 0x1);
            }
            nir_pop_if(b, NULL);
         } else {
            nir_store_var(b, s->hit_kind, tu_rt_triangle_hit_kind(b, rq, true), 0x1);
            tu_rt_store_bary(b, c, rq, true);
         }

         struct tu_rt_sbt_entry chit =
            tu_rt_load_sbt_entry(b, tu_rt_sbt_index(b, c, rq, true), hit_sbt);
         nir_store_var(b, s->shader_record_ptr, chit.shader_record_ptr, 0x1);
         tu_rt_dispatch_recursive(b, c, chit.value, MESA_SHADER_CLOSEST_HIT, payload, 1, cd);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      struct tu_rt_sbt_entry miss = tu_rt_load_sbt_entry(
         b, miss_index,
         offsetof(VkTraceRaysIndirectCommand2KHR, missShaderBindingTableAddress));
      nir_store_var(b, s->shader_record_ptr, miss.shader_record_ptr, 0x1);
      tu_rt_dispatch_recursive(b, c, miss.value, MESA_SHADER_MISS, payload, 1, cd);
   }
   nir_pop_if(b, NULL);
}

static void
tu_rt_execute_callable(nir_builder *b, struct tu_rt_compiler *c,
                       nir_intrinsic_instr *instr)
{
   struct tu_rt_sbt_entry entry = tu_rt_load_sbt_entry(
      b, instr->src[0].ssa,
      offsetof(VkTraceRaysIndirectCommand2KHR, callableShaderBindingTableAddress));
   nir_def *prev_record = nir_load_var(b, c->state.shader_record_ptr);
   nir_store_var(b, c->state.shader_record_ptr, entry.shader_record_ptr, 0x1);
   const struct tu_rt_fn_ctx *ctx = tu_rt_ctx(c, b->impl->function);
   tu_rt_dispatch_recursive(b, c, entry.value, MESA_SHADER_CALLABLE, instr->src[1].ssa,
                            ctx->ray_depth, ctx->call_depth + 1);
   nir_store_var(b, c->state.shader_record_ptr, prev_record, 0x1);
}

struct tu_rt_isec_state {
   struct tu_rt_compiler *compiler;
   nir_shader *ahit;
   unsigned call_depth;
};

static bool
tu_rt_lower_isec_intrinsic(nir_builder *b, nir_intrinsic_instr *instr, void *data)
{
   if (instr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   struct tu_rt_isec_state *isec = (struct tu_rt_isec_state *) data;
   struct tu_rt_compiler *c = isec->compiler;
   struct tu_rt_state *s = &c->state;

   b->cursor = nir_after_instr(&instr->instr);

   nir_def *t = instr->src[0].ssa;
   nir_def *hit_kind = instr->src[1].ssa;

   nir_def *prev_accept = nir_load_var(b, s->accept);
   nir_def *prev_tmax = nir_load_var(b, s->tmax);
   nir_def *prev_hit_kind = nir_load_var(b, s->hit_kind);

   nir_variable *commit =
      nir_local_variable_create(b->impl, glsl_bool_type(), "commit");
   nir_store_var(b, commit, nir_imm_false(b), 0x1);

   nir_def *in_range = nir_iand(b, nir_fge(b, t, nir_load_var(b, s->tmin)),
                                nir_fge(b, nir_load_var(b, s->tmax), t));
   nir_def *terminated = nir_iand(b, nir_load_var(b, s->terminate),
                                  nir_load_var(b, s->accept));
   nir_push_if(b, nir_iand(b, in_range, nir_inot(b, terminated)));
   {
      nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, s->tmax, t, 0x1);
      nir_store_var(b, s->hit_kind, hit_kind, 0x1);

      if (isec->ahit) {
         nir_def *prev_terminate = nir_load_var(b, s->terminate);
         nir_store_var(b, s->terminate, nir_imm_false(b), 0x1);

         nir_push_if(b, nir_inot(b, nir_load_var(b, s->opaque)));
         tu_rt_call_stage(b, c, isec->ahit, nir_load_param(b, 0), 1, isec->call_depth);
         nir_pop_if(b, NULL);

         nir_def *terminate = nir_load_var(b, s->terminate);
         nir_store_var(b, s->terminate, nir_ior(b, terminate, prev_terminate), 0x1);

         nir_push_if(b, terminate);
         nir_jump(b, nir_jump_return);
         nir_pop_if(b, NULL);
      }

      nir_push_if(b, nir_load_var(b, s->accept));
      {
         nir_store_var(b, commit, nir_imm_true(b), 0x1);
      }
      nir_push_else(b, NULL);
      {
         nir_store_var(b, s->accept, prev_accept, 0x1);
         nir_store_var(b, s->tmax, prev_tmax, 0x1);
         nir_store_var(b, s->hit_kind, prev_hit_kind, 0x1);
      }
      nir_pop_if(b, NULL);

      nir_push_if(b, nir_iand(b, nir_load_var(b, commit), nir_load_var(b, s->terminate)));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&instr->def, nir_load_var(b, commit));
   return true;
}

static nir_function *
tu_rt_get_isec_function(nir_builder *b, struct tu_rt_compiler *c, uint32_t group,
                        unsigned call_depth)
{
   uint32_t slot = group * (TU_RT_MAX_CALL_DEPTH + 1) + call_depth;
   if (c->isec_cache[slot])
      return c->isec_cache[slot];

   struct tu_rt_group *g = c->pipeline->groups + group;
   nir_function *function =
      tu_rt_clone_stage(c, b->shader, c->pipeline->stages[g->isec_index]);
   c->isec_cache[slot] = function;
   tu_rt_set_ctx(c, function, MESA_SHADER_INTERSECTION, 1, call_depth);

   struct tu_rt_isec_state isec = {
      .compiler = c,
      .ahit = tu_rt_group_stage(c, g->ahit_index),
      .call_depth = call_depth,
   };
   nir_function_intrinsics_pass(function->impl, tu_rt_lower_isec_intrinsic,
                                nir_metadata_none, &isec);
   return function;
}

static bool
tu_rt_lower_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct tu_rt_compiler *c = (struct tu_rt_compiler *) data;
   struct tu_rt_state *s = &c->state;

   if (instr->type == nir_instr_type_jump) {
      nir_jump_instr *jump = nir_instr_as_jump(instr);
      if (jump->type == nir_jump_halt) {
         jump->type = nir_jump_return;
         return true;
      }
      return false;
   } else if (instr->type != nir_instr_type_intrinsic) {
      return false;
   }

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   nir_def *def = NULL;
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_execute_callable:
      if (tu_rt_ctx(c, b->impl->function)->call_depth < c->max_call_depth)
         tu_rt_execute_callable(b, c, intr);
      break;
   case nir_intrinsic_trace_ray:
      if (tu_rt_ctx(c, b->impl->function)->ray_depth == 0)
         tu_rt_trace_ray(b, c, intr);
      break;
   case nir_intrinsic_ignore_ray_intersection:
      nir_store_var(b, s->accept, nir_imm_false(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_terminate_ray:
      nir_store_var(b, s->accept, nir_imm_true(b), 0x1);
      nir_store_var(b, s->terminate, nir_imm_true(b), 0x1);
      nir_push_if(b, nir_imm_true(b));
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      break;
   case nir_intrinsic_load_ray_launch_id:
      def = nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      def = tu_rt_load_cmd(b, offsetof(VkTraceRaysIndirectCommand2KHR, width), 3);
      break;
   case nir_intrinsic_load_shader_record_ptr:
      def = nir_load_var(b, s->shader_record_ptr);
      break;
   case nir_intrinsic_load_ray_t_min:
      def = nir_load_var(b, s->tmin);
      break;
   case nir_intrinsic_load_ray_t_max:
      def = nir_load_var(b, s->tmax);
      break;
   case nir_intrinsic_load_ray_world_origin:
      def = nir_load_var(b, s->origin);
      break;
   case nir_intrinsic_load_ray_world_direction:
      def = nir_load_var(b, s->dir);
      break;
   case nir_intrinsic_load_ray_object_origin:
      def = nir_load_var(b, s->object_origin);
      break;
   case nir_intrinsic_load_ray_object_direction:
      def = nir_load_var(b, s->object_dir);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      def = nir_load_var(b, s->custom_index);
      break;
   case nir_intrinsic_load_instance_id:
      def = nir_load_var(b, s->instance_id);
      break;
   case nir_intrinsic_load_primitive_id:
      def = nir_load_var(b, s->primitive_id);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      def = nir_load_var(b, s->geometry_index);
      break;
   case nir_intrinsic_load_ray_flags:
      def = nir_load_var(b, s->flags);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      def = nir_load_var(b, s->hit_kind);
      break;
   case nir_intrinsic_load_cull_mask:
      def = nir_iand_imm(b, nir_load_var(b, s->cull_mask), 0xFF);
      break;
   case nir_intrinsic_load_ray_world_to_object:
      def = nir_load_var(b, s->wto[nir_intrinsic_column(intr)]);
      break;
   case nir_intrinsic_load_ray_object_to_world:
      def = nir_load_var(b, s->otw[nir_intrinsic_column(intr)]);
      break;
   default:
      return false;
   }

   if (def)
      nir_def_rewrite_uses(&intr->def, def);
   nir_instr_remove(instr);
   return true;
}

static unsigned
tu_rt_natural_field_offset(const struct glsl_type *st, unsigned field)
{
   unsigned off = 0;
   for (unsigned i = 0; i <= field; i++) {
      unsigned size, al;
      glsl_get_natural_size_align_bytes(glsl_get_struct_field(st, i), &size, &al);
      off = align(off, al);
      if (i == field)
         return off;
      off += size;
   }
   return off;
}

static nir_def *
tu_rt_pun_offset(nir_builder *b, nir_deref_instr *deref)
{
   switch (deref->deref_type) {
   case nir_deref_type_var:
      return nir_imm_int(b, 0);
   case nir_deref_type_cast:
      return tu_rt_pun_offset(b, nir_deref_instr_parent(deref));
   case nir_deref_type_struct: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      return nir_iadd_imm(b, tu_rt_pun_offset(b, parent),
                          tu_rt_natural_field_offset(parent->type, deref->strct.index));
   }
   case nir_deref_type_array:
   case nir_deref_type_ptr_as_array: {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      unsigned size, al;
      glsl_get_natural_size_align_bytes(deref->type, &size, &al);
      unsigned stride = glsl_type_is_vector(parent->type) ? size : align(size, al);
      nir_def *idx = nir_u2u32(b, deref->arr.index.ssa);
      return nir_iadd(b, tu_rt_pun_offset(b, parent), nir_imul_imm(b, idx, stride));
   }
   default:
      UNREACHABLE("unsupported deref in ray tracing pun");
   }
}

static nir_variable *
tu_rt_deref_root_var(nir_deref_instr *deref)
{
   while (deref->deref_type != nir_deref_type_var) {
      deref = nir_deref_instr_parent(deref);
      if (!deref)
         return NULL;
   }
   return deref->var;
}

static void
tu_rt_pun_mismatched_vars(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct hash_table *punned = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_cast ||
             !nir_deref_mode_may_be(deref, nir_var_function_temp))
            continue;
         nir_variable *var = tu_rt_deref_root_var(deref);
         if (!var || _mesa_hash_table_search(punned, var))
            continue;
         unsigned size, al;
         glsl_get_natural_size_align_bytes(var->type, &size, &al);
         nir_variable *words =
            nir_local_variable_create(impl, glsl_array_type(glsl_uint_type(),
                                                            MAX2(DIV_ROUND_UP(size, 4), 1), 4),
                                      "tu_rt_punned");
         _mesa_hash_table_insert(punned, var, words);
      }
   }

   if (!punned->entries) {
      _mesa_hash_table_destroy(punned, NULL);
      return;
   }

   mesa_logw("tu: ray tracing payload type mismatch, punning %u variable(s)",
             punned->entries);

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_safe (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref &&
             intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         nir_variable *root = tu_rt_deref_root_var(deref);
         struct hash_entry *he = root ? _mesa_hash_table_search(punned, root) : NULL;
         if (!he)
            continue;
         nir_variable *words = (nir_variable *) he->data;

         b.cursor = nir_before_instr(instr);
         nir_def *off = tu_rt_pun_offset(&b, deref);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_def *vals[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < intr->def.num_components; c++)
               vals[c] = tu_rt_hw_load_comp(&b, words, off, c, intr->def.bit_size);
            nir_def_replace(&intr->def, nir_vec(&b, vals, intr->def.num_components));
         } else {
            nir_def *v = intr->src[1].ssa;
            unsigned mask = nir_intrinsic_write_mask(intr);
            for (unsigned c = 0; c < v->num_components; c++) {
               if (mask & (1u << c))
                  tu_rt_hw_store_comp(&b, words, off, c, nir_channel(&b, v, c));
            }
            nir_instr_remove(instr);
         }
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   _mesa_hash_table_destroy(punned, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
}

static nir_shader *
tu_rt_build_shader(struct tu_device *dev,
                   struct tu_ray_tracing_pipeline *pipeline,
                   const VkRayTracingPipelineCreateInfoKHR *create_info,
                   VkPipelineCreateFlags2KHR flags)
{
   nir_builder _b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, ir3_get_compiler_options(dev->compiler),
      "tu ray tracing pipeline");
   nir_builder *b = &_b;

   b->shader->info.workgroup_size[0] = 8;
   b->shader->info.workgroup_size[1] = 8;
   b->shader->info.workgroup_size[2] = 1;

   struct tu_rt_compiler c = {};
   c.pipeline = pipeline;
   c.flags = flags;
   c.fn_ctx = _mesa_pointer_hash_table_create(NULL);
   c.fn_cache = (nir_function **)
      calloc(MAX2(pipeline->stage_count, 1) * 2 * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   c.isec_cache = (nir_function **)
      calloc(MAX2(pipeline->group_count, 1) * (TU_RT_MAX_CALL_DEPTH + 1),
             sizeof(nir_function *));
   tu_rt_state_init(b->shader, &c.state);
   c.hit_words = tu_rt_hit_words_var(b->shader);

   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct tu_rt_group *g = pipeline->groups + i;
      if (g->ahit_index != VK_SHADER_UNUSED_KHR)
         c.has_ahit = true;
      if (g->isec_index != VK_SHADER_UNUSED_KHR)
         c.has_isec = true;
   }
   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      c.used |= tu_rt_scan_used(pipeline->stages[i]);

   {
      uint32_t ncall = 0;
      for (uint32_t i = 0; i < pipeline->group_count; i++) {
         nir_shader *s = tu_rt_group_stage(&c, pipeline->groups[i].recursive_index);
         if (s && s->info.stage == MESA_SHADER_CALLABLE &&
             !tu_rt_handle_seen(&c, i, pipeline->groups[i].handle, 0))
            ncall++;
      }
      uint64_t bodies = 0, level = 1;
      c.max_call_depth = 0;
      while (c.max_call_depth < TU_RT_MAX_CALL_DEPTH) {
         level *= MAX2(ncall, 1);
         if (c.max_call_depth >= 2 && bodies + level > 64)
            break;
         bodies += level;
         c.max_call_depth++;
      }
   }

   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *oob = nir_ige(b, nir_channel(b, launch_id, 0), nir_channel(b, launch_size, 0));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 1), nir_channel(b, launch_size, 1)));
   oob = nir_ior(b, oob, nir_ige(b, nir_channel(b, launch_id, 2), nir_channel(b, launch_size, 2)));
   nir_push_if(b, oob);
   nir_jump(b, nir_jump_return);
   nir_pop_if(b, NULL);

   struct tu_rt_sbt_entry raygen = tu_rt_load_sbt_entry(
      b, NULL, offsetof(VkTraceRaysIndirectCommand2KHR, raygenShaderRecordAddress));
   nir_store_var(b, c.state.shader_record_ptr, raygen.shader_record_ptr, 0x1);
   tu_rt_dispatch_recursive(b, &c, raygen.value, MESA_SHADER_RAYGEN,
                            nir_undef(b, 1, 32), 0, 0);

   nir_shader_instructions_pass(b->shader, tu_rt_lower_instr,
                                nir_metadata_none, &c);

   NIR_PASS(_, b->shader, nir_lower_continue_constructs);
   NIR_PASS(_, b->shader, nir_lower_returns);
   NIR_PASS(_, b->shader, nir_inline_functions);
   nir_remove_non_entrypoints(b->shader);
   NIR_PASS(_, b->shader, nir_opt_deref);
   NIR_PASS(_, b->shader, nir_opt_copy_prop);
   NIR_PASS(_, b->shader, nir_opt_deref);

   NIR_PASS(_, b->shader, nir_split_var_copies);
   NIR_PASS(_, b->shader, nir_lower_var_copies);
   tu_rt_pun_mismatched_vars(b->shader);

   NIR_PASS(_, b->shader, nir_lower_global_vars_to_local);
   NIR_PASS(_, b->shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b->shader, nir_opt_copy_prop);
   NIR_PASS(_, b->shader, nir_opt_dce);

   NIR_PASS(_, b->shader, nir_opt_ray_queries);
   NIR_PASS(_, b->shader, tu_nir_lower_ray_queries);
   NIR_PASS(_, b->shader, nir_split_var_copies);
   NIR_PASS(_, b->shader, nir_lower_var_copies);
   NIR_PASS(_, b->shader, nir_lower_system_values);

   ralloc_free(c.fn_ctx);
   free(c.fn_cache);
   free(c.isec_cache);

   nir_shader_gather_info(b->shader, nir_shader_get_entrypoint(b->shader));

   struct ir3_optimize_options optimize_options = {};
   ir3_optimize_loop(dev->compiler, &optimize_options, b->shader);

   return b->shader;
}

static VkResult
tu_rt_pipeline_create(struct tu_device *dev,
                      struct vk_pipeline_cache *cache,
                      const VkRayTracingPipelineCreateInfoKHR *create_info,
                      VkPipelineCreateFlags2KHR flags,
                      const VkAllocationCallbacks *alloc,
                      VkPipeline *out_pipeline)
{
   VK_FROM_HANDLE(tu_pipeline_layout, layout, create_info->layout);
   VkResult result;

   struct tu_ray_tracing_pipeline *pipeline =
      (struct tu_ray_tracing_pipeline *) vk_object_zalloc(
         &dev->vk, alloc, sizeof(*pipeline), VK_OBJECT_TYPE_PIPELINE);
   if (!pipeline)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   pipeline->base.base.type = TU_PIPELINE_RAY_TRACING;
   pipeline->base.base.executables_mem_ctx = ralloc_context(NULL);
   util_dynarray_init(&pipeline->base.base.executables,
                      pipeline->base.base.executables_mem_ctx);
   pipeline->base.base.active_stages = TU_RT_STAGES;
   pipeline->mem_ctx = ralloc_context(NULL);

   pipeline->stage_count = create_info->stageCount;
   pipeline->group_count = create_info->groupCount;
   if (create_info->pLibraryInfo) {
      for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
         VK_FROM_HANDLE(tu_pipeline, lib_base,
                        create_info->pLibraryInfo->pLibraries[l]);
         struct tu_ray_tracing_pipeline *lib =
            tu_pipeline_to_ray_tracing(lib_base);
         pipeline->stage_count += lib->stage_count;
         pipeline->group_count += lib->group_count;
      }
   }

   pipeline->stages = rzalloc_array(pipeline->mem_ctx, nir_shader *,
                                    MAX2(pipeline->stage_count, 1));
   pipeline->stage_hashes = rzalloc_array(pipeline->mem_ctx, uint8_t,
                                          MAX2(pipeline->stage_count, 1) * BLAKE3_KEY_LEN);
   pipeline->groups = rzalloc_array(pipeline->mem_ctx, struct tu_rt_group,
                                    MAX2(pipeline->group_count, 1));

   struct tu_shader_key key = {};
   tu_shader_key_subgroup_size(&key, false, false, NULL, dev);
   struct vk_pipeline_robustness_state rs;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &rs,
                                     create_info->pNext, NULL);
   tu_shader_key_robustness(&key, &rs);
   key.fast_math = debug_get_bool_option("TU_RT_FAST_MATH", false);

   {
      uint32_t s = 0;
      for (; s < create_info->stageCount; s++)
         vk_pipeline_hash_shader_stage(flags, create_info->pStages + s, &rs,
                                       pipeline->stage_hashes + s * BLAKE3_KEY_LEN);
      if (create_info->pLibraryInfo) {
         for (uint32_t l = 0; l < create_info->pLibraryInfo->libraryCount; l++) {
            VK_FROM_HANDLE(tu_pipeline, lib_base,
                           create_info->pLibraryInfo->pLibraries[l]);
            struct tu_ray_tracing_pipeline *lib = tu_pipeline_to_ray_tracing(lib_base);
            memcpy(pipeline->stage_hashes + s * BLAKE3_KEY_LEN, lib->stage_hashes,
                   lib->stage_count * BLAKE3_KEY_LEN);
            s += lib->stage_count;
         }
      }
   }

   tu_rt_init_groups(dev, pipeline, create_info);

   if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      result = tu_rt_compile_stages(dev, cache, pipeline, create_info, flags, &key);
      if (result != VK_SUCCESS)
         goto fail;
   } else {
      static const char version[] = "tu-rt-3";
      unsigned char blake3[BLAKE3_KEY_LEN];
      struct mesa_blake3 ctx;
      _mesa_blake3_init(&ctx);
      _mesa_blake3_update(&ctx, version, sizeof(version));
      if (layout)
         _mesa_blake3_update(&ctx, layout->blake3, sizeof(layout->blake3));
      _mesa_blake3_update(&ctx, &key, sizeof(key));
      enum ir3_shader_debug ir3_debug_key = ir3_shader_debug_hash_key();
      _mesa_blake3_update(&ctx, &ir3_debug_key, sizeof(ir3_debug_key));
      VkPipelineCreateFlags2KHR hflags = flags &
         (VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR |
          VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR);
      _mesa_blake3_update(&ctx, &hflags, sizeof(hflags));
      uint32_t depth = MIN2(create_info->maxPipelineRayRecursionDepth, 1);
      _mesa_blake3_update(&ctx, &depth, sizeof(depth));
      for (uint32_t g = 0; g < pipeline->group_count; g++) {
         const struct tu_rt_group *grp = pipeline->groups + g;
         _mesa_blake3_update(&ctx, &grp->handle, sizeof(grp->handle));
         uint32_t idx[3] = { grp->recursive_index, grp->ahit_index, grp->isec_index };
         for (unsigned k = 0; k < 3; k++) {
            if (idx[k] == VK_SHADER_UNUSED_KHR) {
               uint32_t none = ~0u;
               _mesa_blake3_update(&ctx, &none, sizeof(none));
            } else {
               _mesa_blake3_update(&ctx, pipeline->stage_hashes + idx[k] * BLAKE3_KEY_LEN,
                                   BLAKE3_KEY_LEN);
            }
         }
      }
      _mesa_blake3_final(&ctx, blake3);

      struct tu_shader *shader = NULL;
      nir_shader *nir = NULL;
      if (!(flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR))
         shader = tu_compute_pipeline_cache_lookup(dev, cache, blake3);
      if (!shader) {
         if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) {
            result = VK_PIPELINE_COMPILE_REQUIRED;
            goto fail;
         }
         result = tu_rt_compile_stages(dev, cache, pipeline, create_info, flags, &key);
         if (result != VK_SUCCESS)
            goto fail;
         nir = tu_rt_build_shader(dev, pipeline, create_info, flags);
         ralloc_steal(pipeline->mem_ctx, nir);
      }

      result = tu_compute_pipeline_init_nir(dev, &pipeline->base, layout, cache,
                                            shader, nir, &key, blake3);
      if (result != VK_SUCCESS)
         goto fail;
   }

   *out_pipeline = tu_pipeline_to_handle(&pipeline->base.base);
   return VK_SUCCESS;

fail:
   tu_pipeline_finish(&pipeline->base.base, dev, alloc);
   vk_object_free(&dev->vk, alloc, pipeline);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_CreateRayTracingPipelinesKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                VkPipelineCache pipelineCache,
                                uint32_t createInfoCount,
                                const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
                                const VkAllocationCallbacks *pAllocator,
                                VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(tu_device, dev, _device);
   VK_FROM_HANDLE(vk_pipeline_cache, cache, pipelineCache);
   VkResult result = VK_SUCCESS;

   uint32_t i = 0;
   for (; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags =
         vk_rt_pipeline_create_flags(&pCreateInfos[i]);
      VkResult r = tu_rt_pipeline_create(dev, cache, &pCreateInfos[i], flags,
                                         pAllocator, &pPipelines[i]);
      if (r != VK_SUCCESS) {
         result = r;
         pPipelines[i] = VK_NULL_HANDLE;
         if (flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
            break;
      }
   }

   for (; i < createInfoCount; i++)
      pPipelines[i] = VK_NULL_HANDLE;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingShaderGroupHandlesKHR(VkDevice device,
                                      VkPipeline _pipeline,
                                      uint32_t firstGroup,
                                      uint32_t groupCount,
                                      size_t dataSize,
                                      void *pData)
{
   VK_FROM_HANDLE(tu_pipeline, base, _pipeline);
   struct tu_ray_tracing_pipeline *pipeline = tu_pipeline_to_ray_tracing(base);

   uint8_t *data = (uint8_t *) pData;
   memset(data, 0, dataSize);
   for (uint32_t i = 0; i < groupCount; i++) {
      memcpy(data + i * TU_RT_GROUP_HANDLE_SIZE,
             &pipeline->groups[firstGroup + i].handle, sizeof(uint32_t));
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
tu_GetRayTracingCaptureReplayShaderGroupHandlesKHR(VkDevice device,
                                                   VkPipeline pipeline,
                                                   uint32_t firstGroup,
                                                   uint32_t groupCount,
                                                   size_t dataSize,
                                                   void *pData)
{
   memset(pData, 0, dataSize);
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
tu_GetRayTracingShaderGroupStackSizeKHR(VkDevice device,
                                        VkPipeline pipeline,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR groupShader)
{
   return 4;
}

VKAPI_ATTR void VKAPI_CALL
tu_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
}
