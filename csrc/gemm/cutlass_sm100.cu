// ================================================================
// FlashRT — CUTLASS FP8 GEMM implementations for SM100/SM110
//
// Tile configurations tuned for the Pi0.5 / Pi0 / GROOT shape mix:
//   - SqGemm (256×256×128): QKV, O projection
//   - T1Gemm (128×256×128, 2SM): Gate, Up (split for L2)
//   - WideGemm (256×128×128): Down projection
//   - PlainGemm (256×128×64): General fallback
//   - GeluGemm (256×128×64 + GELU): SigLIP FFN Up
//
// All use FP8 E4M3 inputs → FP16 output (matching Thor pipeline).
// Weight B is ColumnMajor (stored as [K, N] row-major in memory).
// ================================================================

#include "gemm_types_sm100.h"
#include "cutlass/util/device_memory.h"
#include <cuda_runtime.h>
#include <cstdio>

// ── Generic runner: initialize + run on given stream ──
template <typename GemmOp>
static int cutlass_run_impl(void* A, void* B, void* D,
                             int M, int N, int K,
                             float alpha, float beta,
                             cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;

    // CUTLASS stride computation
    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},  // problem size
        {(ElementA*)A, stride_A, (ElementB*)B, stride_B},
        {{alpha, beta}, (ElementD*)D, stride_D, (ElementD*)D, stride_D}
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    return 0;
}

static int cutlass_fp8_wide_geglu_fp8out_impl(void* A, void* B_up, void* gate_aux, void* D,
                                               int M, int N, int K,
                                               float alpha, const float* act_scale,
                                               cudaStream_t stream) {
    using GemmOp = sm100_wide_geglu_fp8out::Gemm;
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;
    using ElementD = typename GemmOp::ElementD;
    using StrideAux = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});
    auto stride_Aux = cutlass::make_cute_packed_stride(StrideAux{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)A, stride_A, (ElementB*)B_up, stride_B},
        {
            {alpha, 0.0f, nullptr, nullptr, {}, {}, {act_scale},
             (cutlass_fp16*)gate_aux, stride_Aux},
            (ElementC*)D, stride_C, (ElementD*)D, stride_D
        }
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] geglu cannot implement: M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
        return -1;
    }
    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] geglu init failed: M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
        return -2;
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] geglu run failed: M=%d N=%d K=%d status=%d\n", M, N, K, int(status));
        return -3;
    }
    return 0;
}

static int cutlass_fp8_siglip_gelu_fp8out_impl(void* A, void* B, void* D,
                                                void* bias, int M, int N, int K,
                                                float alpha, cudaStream_t stream) {
    using GemmOp = sm100_siglip_gelu_fp8out::Gemm;
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;
    using ElementD = typename GemmOp::ElementD;

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)A, stride_A, (ElementB*)B, stride_B},
        {{alpha, 0.0f}, (ElementC*)D, stride_C, (ElementD*)D, stride_D}
    };
    args.epilogue.thread.bias_ptr = (cutlass_fp16 const*)bias;

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS] SigLIP GELU cannot implement: M=%d N=%d K=%d status=%d\n",
                M, N, K, int(status));
        return -1;
    }
    status = gemm.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = gemm.run(stream);
    return status == cutlass::Status::kSuccess ? 0 : -3;
}

// ── Exported C functions ──
extern "C" {

int cutlass_fp8_sq(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_t1(void* A, void* B, void* D, int M, int N, int K,
                    float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_t1::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_geglu_fp8out(void* A, void* B_up, void* gate_aux, void* D,
                                   int M, int N, int K, float alpha,
                                   const float* act_scale, cudaStream_t stream) {
    return cutlass_fp8_wide_geglu_fp8out_impl(A, B_up, gate_aux, D, M, N, K,
                                               alpha, act_scale, stream);
}

int cutlass_fp8_siglip_gelu_fp8out(void* A, void* B, void* D, void* bias,
                                    int M, int N, int K, float alpha,
                                    cudaStream_t stream) {
    return cutlass_fp8_siglip_gelu_fp8out_impl(
        A, B, D, bias, M, N, K, alpha, stream);
}

int cutlass_fp8_plain(void* A, void* B, void* D, int M, int N, int K,
                       float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_plain::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_gelu(void* A, void* B, void* D, int M, int N, int K,
                      float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_gelu::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

// FP32 output variants — for models with activations exceeding FP16 range
int cutlass_fp8_sq_f32out(void* A, void* B, void* D, int M, int N, int K,
                           float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq_f32out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_f32out(void* A, void* B, void* D, int M, int N, int K,
                             float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide_f32out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

// BF16 output variants
int cutlass_fp8_sq_bf16out(void* A, void* B, void* D, int M, int N, int K,
                            float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_sq_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_wide_bf16out(void* A, void* B, void* D, int M, int N, int K,
                              float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_wide_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

int cutlass_fp8_t1_bf16out(void* A, void* B, void* D, int M, int N, int K,
                            float alpha, float beta, cudaStream_t stream) {
    return cutlass_run_impl<sm100_t1_bf16out::Gemm>(A, B, D, M, N, K, alpha, beta, stream);
}

}  // extern "C"
