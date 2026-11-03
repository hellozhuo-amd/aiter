// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

void add_rmsnorm_quant(aiter_tensor_t& out,
                       aiter_tensor_t& input,
                       aiter_tensor_t& residual_in,
                       aiter_tensor_t& residual_out,
                       aiter_tensor_t& scale,
                       aiter_tensor_t& weight,
                       double epsilon,
                       int group_size     = 0,
                       bool shuffle_scale = false,
                       bool gemma_norm    = false);

void add_rmsnorm(aiter_tensor_t& out,
                 aiter_tensor_t& input,
                 aiter_tensor_t& residual_in,
                 aiter_tensor_t& residual_out,
                 aiter_tensor_t& weight,
                 double epsilon,
                 bool gemma_norm = false);

void rmsnorm_quant(aiter_tensor_t& out,
                   aiter_tensor_t& input,
                   aiter_tensor_t& scale,
                   aiter_tensor_t& weight,
                   double epsilon,
                   int group_size     = 0,
                   bool shuffle_scale = false,
                   bool gemma_norm    = false);

void rmsnorm(aiter_tensor_t& out,
             aiter_tensor_t& input,
             aiter_tensor_t& weight,
             double epsilon,
             bool gemma_norm = false);

// out = (rmsnorm(input, weight) + residual) * scalar[0], bitwise equal to rmsnorm() followed by the fp32
// elementwise (h + residual) * scalar rounded once to nearest even.
void rmsnorm_post_add_scale(aiter_tensor_t& out,
                            aiter_tensor_t& input,
                            aiter_tensor_t& residual,
                            aiter_tensor_t& weight,
                            aiter_tensor_t& scalar,
                            double epsilon,
                            bool gemma_norm = false);

} // namespace aiter
