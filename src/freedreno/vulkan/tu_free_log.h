/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_FREE_LOG_H
#define TU_FREE_LOG_H

#include "tu_knl.h"

extern bool tu_free_log_on;

void tu_free_log_device_init(void);
void tu_free_log_device_finish(void);

void tu_free_log_bo_alloc(struct tu_bo *bo, uint32_t flags, const char *name);
void tu_free_log_bo_free(const struct tu_bo *bo);
void tu_free_log_vma(const char *op, uint64_t iova, uint64_t size,
                     uint32_t flags);
void tu_free_log_free_memory(const void *mem, uint64_t iova, uint64_t size,
                             const struct tu_bo *bo, bool lazy);
void tu_free_log_destroy_buffer(const void *buffer, uint64_t va,
                                uint64_t size, bool sparse);
void tu_free_log_create_as(const void *as, uint64_t va, uint64_t size,
                           uint32_t type);
void tu_free_log_destroy_as(const void *as, uint64_t addr, uint64_t size);

#endif
