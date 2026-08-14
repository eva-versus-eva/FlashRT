// 优化：FP8 Gate/Up 在同一 CTA 内复用 X tile，并通过 SMEM 传递 FP16 gate。
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#include "gemm_types_sm100.h"
#include "sm100_smem_aux_visitor.hpp"
#include "flashrt_megakernel_geglu_kernel.hpp"

#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;
using fp8_t = cutlass::float_e4m3_t;
using fp16_t = cutlass::half_t;

namespace {

using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _2, _1>;

// 优化：phase1 gate 只写 SMEM 供 phase2 使用，跳过无用的 global D store。
using FusionGate = flashrt::megakernel::fusion::LinCombEltActSmemAuxStore<
    cutlass::epilogue::thread::Identity, fp8_t, float, fp16_t,
    fp8_t, float, 16, cutlass::FloatRoundStyle::round_to_nearest, true>;

using FusionUp = flashrt::megakernel::fusion::LinCombDeEltActSmemAuxLoad<
    GeGluMulScale, fp8_t, float, fp16_t>;

using CollectiveEpiGate = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp8_t, cutlass::layout::RowMajor, 16,
    fp8_t, cutlass::layout::RowMajor, 16,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionGate>::CollectiveOp;

using CollectiveEpiUp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, fp8_t, cutlass::layout::RowMajor, 16,
    fp8_t, cutlass::layout::RowMajor, 16,
    cutlass::epilogue::collective::EpilogueScheduleAuto, FusionUp>::CollectiveOp;

using CollectiveMmaGate = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp8_t, cutlass::layout::RowMajor, 16,
    fp8_t, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using CollectiveMmaUp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    fp8_t, cutlass::layout::RowMajor, 16,
    fp8_t, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::FlashRtMegakernelGeGLUFusedGemm<
    Shape<int, int, int, int>,
    CollectiveMmaGate, CollectiveEpiGate,
    CollectiveMmaUp, CollectiveEpiUp,
    void>;

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

}  // namespace

extern "C" int flashrt_megakernel_geglu_fp8(
    void* X, void* W_gate, void* W_up, void* gate_scratch, void* hidden,
    int M, int N, int K, float alpha, const float* act_scale,
    cudaStream_t stream) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementD = typename GemmOp::ElementD;
    using ElementD2 = typename CollectiveEpiUp::ElementD;

    auto sA = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideA{}, {M, K, 1});
    auto sB = cutlass::make_cute_packed_stride(
        typename GemmOp::GemmKernel::StrideB{}, {N, K, 1});
    auto sGate = cutlass::make_cute_packed_stride(
        typename CollectiveEpiGate::StrideD{}, {M, N, 1});
    auto sHidden = cutlass::make_cute_packed_stride(
        typename CollectiveEpiUp::StrideD{}, {M, N, 1});

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(ElementA*)X, sA, (ElementB*)W_gate, sB},
        {
            {alpha, 0.0f, nullptr, nullptr, {}, {}, {}},
            nullptr, {}, (ElementD*)gate_scratch, sGate
        },
        {(ElementA*)X, sA, (ElementB*)W_up, sB},
        {
            {alpha, 0.0f, nullptr, nullptr, {}, {}, {act_scale}},
            nullptr, {}, (ElementD2*)hidden, sHidden
        }
    };

    GemmOp gemm;
    size_t ws_size = GemmOp::get_workspace_size(args);
    static cutlass::device_memory::allocation<uint8_t> workspace(0);
    if (ws_size > workspace.size()) {
        workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        fprintf(stderr, "[flashrt_megakernel_geglu_fp8] cannot implement M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    if (gemm.initialize(args, workspace.get(), stream) != cutlass::Status::kSuccess) {
        return -2;
    }
    return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}
