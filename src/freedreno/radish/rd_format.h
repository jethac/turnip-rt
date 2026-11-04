/* a5xx format tables (from Freedreno's fd5_format.c). SPDX-License-Identifier: MIT */
#ifndef RD_FORMAT_H
#define RD_FORMAT_H
#include "util/format/u_formats.h"
#include "common/fd5_hw.h"
enum a5xx_vtx_fmt rd5_pipe2vtx(enum pipe_format format);
enum a5xx_tex_fmt rd5_pipe2tex(enum pipe_format format);
enum a5xx_color_fmt rd5_pipe2color(enum pipe_format format);
enum a3xx_color_swap rd5_pipe2swap(enum pipe_format format);
enum a5xx_depth_format rd5_pipe2depth(enum pipe_format format);
#endif
