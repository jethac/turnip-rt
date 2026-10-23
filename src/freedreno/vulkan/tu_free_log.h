/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_FREE_LOG_H
#define TU_FREE_LOG_H

#include "tu_knl.h"

extern bool tu_free_log_on;
extern bool tu_rt_label_on;

void tu_free_log_device_init(void);
void tu_free_log_device_finish(void);

void tu_free_log_bo_alloc(struct tu_bo *bo, uint32_t flags, const char *name);
void tu_free_log_bo_free(const struct tu_bo *bo);
void tu_free_log_config(bool epw_broad, bool cs_isz_wa,
                        uint32_t cs_isz_restore);
void tu_free_log_cs_isz(const char *bin, uint32_t instrlen, bool wa,
                        uint32_t restore, uint32_t cond);
void tu_free_log_vma(const char *op, uint64_t iova, uint64_t size,
                     uint32_t flags);
void tu_free_log_free_memory(const void *mem, uint64_t iova, uint64_t size,
                             const struct tu_bo *bo, bool lazy);
void tu_free_log_destroy_buffer(const void *buffer, uint64_t va,
                                uint64_t size, bool sparse);
void tu_free_log_create_as(const void *as, uint64_t va, uint64_t size,
                           uint32_t type);
void tu_free_log_destroy_as(const void *as, uint64_t addr, uint64_t size);

void tu_free_log_rt_stage(uint32_t handle, const char *stage,
                          const uint32_t *spirv, size_t size,
                          const char *entry);
void tu_free_log_rt_pipeline(const void *pipeline, bool library,
                             uint32_t stage_count, uint32_t group_count,
                             uint32_t variant_count);
void tu_free_log_rt_variant(const void *pipeline, uint32_t index,
                            uint32_t count, uint32_t handle,
                            const char *blake3);
uint32_t tu_free_log_rt_dispatch(const void *cmd, const void *pipeline,
                                 uint32_t variant_count, const uint32_t *dims,
                                 uint64_t cmd_va, uint64_t raygen_va);

#endif
