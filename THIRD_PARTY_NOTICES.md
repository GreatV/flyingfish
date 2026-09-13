# Third-party notices

`crates/ff-music/src/` contains Rust adaptations of the MiniMax Music3 model
and pipeline algorithms in Hugging Face Diffusers.
Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
These adaptations change the execution backend to Candle and stream projection
weights; they retain the upstream Apache-2.0 license, included in [LICENSE](LICENSE).

`crates/ff-minicpm/src/dspark.rs` adapts the DSpark/DFlash architecture and
verification contract from [SGLang](https://github.com/sgl-project/sglang),
Copyright 2023-2024 SGLang Team, under Apache-2.0. The implementation uses
Candle projections and a local KV cache instead of SGLang's execution runtime.

The CLIP dual-encoder implementation follows Transformers'
CLIP architecture, Copyright 2021 The OpenAI Team Authors and The HuggingFace
Team, under Apache-2.0. The Rust vision path streams weights through Candle.

The DINOv2/DINOv3 conditioners follow the Transformers implementations under
Apache-2.0. DINOv2: Copyright 2024 Meta Inc. and the HuggingFace Inc. team.
DINOv3: Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
Downloaded pretrained DINO weights retain their original model licenses and
are not redistributed in this repository.

The following Flyingfish sources contain specialized transcriptions of CUDA
algorithms from PyTorch:

- `src/cuda/h3_rms_norm_bf16.cu`, from
  `aten/src/ATen/native/cuda/layer_norm_kernel.cu` and
  `aten/src/ATen/native/cuda/thread_constants.h`;
- `src/cuda/h3_sdpa_softmax_f32.cu`, from
  `aten/src/ATen/native/cuda/PersistentSoftmax.cuh` and
  `aten/src/ATen/native/cuda/SoftMax.cu`, including the block reduction order
  defined by `aten/src/ATen/native/cuda/block_reduce.cuh`;
- `src/cuda/qwen_layer_norm_bf16.cu`, from
  `aten/src/ATen/native/cuda/layer_norm_kernel.cu`,
  `aten/src/ATen/native/cuda/thread_constants.h`, and
  `c10/cuda/CUDAMathCompat.h`;
- `src/cuda/qwen_gelu_bf16.cu`, from
  `aten/src/ATen/native/cuda/ActivationGeluKernel.cu` and
  `c10/cuda/CUDAMathCompat.h`; the checked-in
  `src/cuda/qwen_gelu_bf16_nvrtc130.ptx` is generated from that transcription
  by `scripts/compile-qwen-gelu-nvrtc.py` because the pinned official Torch
  wheel's CUDA 13.0 `erf` implementation is part of the numerical contract;
- `src/cuda/qwen_attention_scale_bf16.cu`, from
  `aten/src/ATen/native/cuda/BinaryMulKernel.cu` and
  `aten/src/ATen/native/cuda/Loops.cuh`;
- `src/cuda/qwen_head_mean_f32.cu`, from
  `aten/src/ATen/native/cuda/ReduceMomentKernel.cu`,
  `aten/src/ATen/native/cuda/Reduce.cuh`, and
  `aten/src/ATen/native/SharedReduceOps.h`.

PyTorch's BSD-style license notice follows.

## PyTorch

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## NVIDIA CUDA and cuDNN runtime libraries

The CUDA execution paths dynamically use external NVIDIA runtime libraries,
including cuBLAS, cuBLASLt, and cuDNN. These binary libraries are not source
files in this repository and are not relicensed under Flyingfish's Apache-2.0
license or PyTorch's BSD-style license. Their use and any redistribution of a
binary or container that includes them remain governed by NVIDIA's applicable
[CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) and
[cuDNN Software License Agreement](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html).
The cuDNN agreement identifies runtime `.so` and `.dll` files as distributable
under that agreement; distributors must independently satisfy its current
conditions.

## TRELLIS and TRELLIS.2

The sparse operators, flow models, Gaussian decoding and dual-grid mesh/VAEs
in `crates/ff-trellis` are Rust adaptations of Microsoft TRELLIS and TRELLIS.2.
The implementation changes the execution backend to Candle, uses bounded
im2col/attention work, and exposes local PLY output. Upstream's MIT notice follows.

```text
MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```

`third_party/candle-kernels/` is a vendored copy of `candle-kernels` 0.11.0
from the Candle project, redirected into this workspace through
`[patch.crates-io]`.
Copyright (c) 2023 Hugging Face. Licensed under MIT OR Apache-2.0; this
workspace uses it under Apache-2.0, the same licence it carries here.

The copy is unmodified except for one guard in `src/compatibility.cuh`. That
file supplies `__hmax_nan` and `__hmin_nan` for targets below sm_80, which the
toolkit used to declare only for sm_80 and later. CUDA 12.6 changed that:
`cuda_fp16.hpp` now declares both for every architecture and selects the
implementation internally, so the upstream block is a redefinition and any
pre-Ampere build fails. The vendored copy adds `&& CUDA_VERSION < 12060` to
that one `#if`, and includes `<cuda.h>` because the fp16/bf16/fp8 headers do
not define `CUDA_VERSION` in a device-only compilation.

Compiled from a single path, so that nvcc's file-derived symbol prefixes and
embedded `__FILE__` strings are held constant, all twelve kernels emit
byte-identical PTX at `compute_89` before and after the change. At
`compute_75` the upstream copy compiles two of twelve and the vendored copy
compiles twelve.
