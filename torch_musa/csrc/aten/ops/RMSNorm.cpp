#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/layer_norm.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

namespace {

ScalarType InferInvvarDType(ScalarType input_dtype) {
  ScalarType invvar_dtype = input_dtype;
  switch (input_dtype) {
    case ScalarType::Half:
    case ScalarType::BFloat16:
    case ScalarType::Float:
      invvar_dtype = ScalarType::Float;
      break;
    default:
      invvar_dtype = input_dtype;
  }
  return invvar_dtype;
}

Tensor CreateRMSNormFwdInvvar(
    const Tensor& input,
    IntArrayRef normalized_shape) {
  const auto input_shape = input.sizes();
  const auto invvar_ndim = input_shape.size() - normalized_shape.size();
  const auto invvar_dtype = InferInvvarDType(input.scalar_type());

  return at::empty(
      IntArrayRef(input_shape.data(), invvar_ndim),
      input.options()
          .dtype(invvar_dtype)
          .memory_format(MemoryFormat::Contiguous));
}

std::vector<int> CreateMuDNNNormAxes(
    const Tensor& input,
    IntArrayRef normalized_shape) {
  const auto norm_ndim = normalized_shape.size();
  std::vector<int> norm_axes(norm_ndim);
  std::iota(
      norm_axes.begin(),
      norm_axes.end(),
      static_cast<int>(input.dim()) - static_cast<int>(norm_ndim));
  return norm_axes;
}

void CheckDevice(
    const char* dst_name,
    Device dst,
    const char* src_name,
    const Tensor& src,
    const char* op) {
  TORCH_CHECK(
      !src.defined() || src.device() == dst,
      "Devices of ",
      dst_name,
      "/",
      src_name,
      " tensors of ",
      op,
      " must be the same, but now ",
      dst_name,
      " in ",
      dst,
      ", ",
      src_name,
      " in ",
      src.device());
}

void CheckDType(
    const char* dst_name,
    ScalarType dst,
    const char* src_name,
    const Tensor& src,
    const char* op) {
  TORCH_CHECK(
      !src.defined() || src.scalar_type() == dst,
      "DTypes of ",
      dst_name,
      "/",
      src_name,
      " tensors of ",
      op,
      " must be the same, but now ",
      dst_name,
      " in ",
      dst,
      ", ",
      src_name,
      " in ",
      src.scalar_type());
}

void FusedRMSNormForwardCall(
    const Tensor& contig_input,
    const Tensor& contig_gamma,
    IntArrayRef normalized_shape,
    double eps,
    Tensor& contig_output,
    Tensor& contig_invvar) {
  if (C10_UNLIKELY(contig_output.numel() == 0)) {
    return;
  }

  ::musa::dnn::RMSNorm op;
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  const auto norm_axes = CreateMuDNNNormAxes(contig_input, normalized_shape);
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axes.size(), norm_axes.data()), "SetAxis");
  auto mudnn_out = CreateMUTensor(contig_output);
  auto mudnn_invvar = CreateMUTensor(contig_invvar);
  auto mudnn_in = CreateMUTensor(contig_input);
  auto mudnn_gamma = contig_gamma.defined() ? CreateMUTensor(contig_gamma)
                                            : CreateEmptyMUTensor();
  muHandle& h = GetMudnnHandle();
  CHECK_MUDNN_STATUS(
      op.Run(h, mudnn_out, mudnn_invvar, mudnn_in, mudnn_gamma),
      "RunRMSNormFwd");
}

} // anonymous namespace

std::tuple<Tensor, Tensor> FusedRMSNormForward(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt,
    std::optional<double> eps) {
  const auto weight = weight_opt.value_or(Tensor());
  (void)native::_check_layer_norm_inputs(
      input, normalized_shape, weight, Tensor());

  const auto input_device = input.device();
  TORCH_CHECK(
      input_device.is_privateuseone(),
      "Device of input tensor of RMSNorm must be MUSA, but now is ",
      input_device);
  CheckDevice("input", input_device, "weight", weight, "FusedRMSNormForward");

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == ScalarType::Float || input_dtype == ScalarType::Half ||
          input_dtype == ScalarType::BFloat16,
      "Dtype of input tensor of RMSNorm only supports Float32, Half and BFloat16, but now is ",
      input_dtype);
  CheckDType("input", input_dtype, "weight", weight, "FusedRMSNormForward");

  const c10::musa::MUSAGuard device_guard(input_device);
  const Tensor contig_input = (input.dim() == 2 && input.stride(1) == 1)
      ? input
      : FormatContiguous(input, MemoryFormat::Contiguous);
  const auto contig_gamma = weight.defined()
      ? FormatContiguous(weight, MemoryFormat::Contiguous)
      : Tensor();
  auto contig_output = at::empty_like(
      contig_input,
      contig_input.options().memory_format(MemoryFormat::Contiguous));
  auto contig_invvar = CreateRMSNormFwdInvvar(contig_input, normalized_shape);
  auto acc_type = at::toAccumulateType(input_dtype, /*is_cuda=*/true);
  double eps_val = eps.has_value();
  if (acc_type == at::ScalarType::Float) {
    eps_val = eps.value_or(std::numeric_limits<float>::epsilon());
  } else {
    eps_val = eps.value_or(std::numeric_limits<double>::epsilon());
  }

  FusedRMSNormForwardCall(
      contig_input,
      contig_gamma,
      normalized_shape,
      eps_val,
      contig_output,
      contig_invvar);
  return {contig_output, contig_invvar};
}

std::tuple<Tensor, Tensor> FusedRMSNormBackward(
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt,
    std::array<bool, 2> grad_input_mask) {
  const c10::musa::MUSAGuard device_guard(input.device());
  UNUSED(grad_input_mask); // TODO(@ai-infra)
  const auto weight = weight_opt.value_or(Tensor());
  const auto contig_input = FormatContiguous(input, MemoryFormat::Contiguous);
  const auto contig_grad_output =
      FormatContiguous(grad_out, MemoryFormat::Contiguous);
  const auto contig_rstd = FormatContiguous(rstd, MemoryFormat::Contiguous);

  auto contig_grad_input = at::empty_like(contig_input);
  const bool has_weight = weight.defined();
  auto contig_gamma = has_weight
      ? FormatContiguous(weight, MemoryFormat::Contiguous)
      : Tensor();
  auto contig_grad_gamma = has_weight ? at::empty_like(contig_gamma) : Tensor();

  ::musa::dnn::RMSNorm op;
  const auto norm_axes = CreateMuDNNNormAxes(contig_input, normalized_shape);
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axes.size(), norm_axes.data()), "SetAxis");
  auto mudnn_dx = CreateMUTensor(contig_grad_input);
  auto mudnn_dg =
      has_weight ? CreateMUTensor(contig_grad_gamma) : CreateEmptyMUTensor();
  auto mudnn_dy = CreateMUTensor(contig_grad_output);
  auto mudnn_in = CreateMUTensor(contig_input);
  auto mudnn_rstd = CreateMUTensor(contig_rstd);
  auto mudnn_gamma =
      has_weight ? CreateMUTensor(contig_gamma) : CreateEmptyMUTensor();
  muHandle& h = GetMudnnHandle();
  CHECK_MUDNN_STATUS(
      op.RunBwd(
          h,
          mudnn_dx,
          mudnn_dg,
          mudnn_dy,
          mudnn_in,
          mudnn_rstd,
          mudnn_gamma,
          InternalMemAlloc),
      "RunRMSNormBwd");

  return {contig_grad_input, contig_grad_gamma};
}

} // namespace musa
} // namespace at
