/*
 * Copyright © 2026 Jetha Chan
 * SPDX-License-Identifier: MIT
 */

#ifndef IR3_REPLAY_H
#define IR3_REPLAY_H

#include "ir3_compiler.h"
#include "ir3_shader.h"

#define IR3_REPLAY_MAGIC 0x44335249u
#define IR3_REPLAY_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

void ir3_replay_dump(const char *dir, const struct ir3_compiler *compiler,
                     const nir_shader *nir,
                     const struct ir3_shader_options *options,
                     const struct ir3_shader_key *key);

#ifdef __cplusplus
}
#endif

#endif
