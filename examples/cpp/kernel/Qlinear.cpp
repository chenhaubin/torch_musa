#include "Qlinear.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/matmul.h>
#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
// Numerics are the same as conv (see aten/src/ATen/native/quantized/Conv.cpp):
namespace at {
namespace native {
DEFINE_DISPATCH(matmul_stub);
REGISTER_NO_CPU_DISPATCH(matmul_stub);
} // namespace native
namespace musa {
static void quantized_matmul_per_channel(
    at::Tensor mat1, // act
    double input_scale,
    int64_t input_zero_point,
    at::Tensor mat2, // weight
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<Tensor> bias,
    at::Tensor& result, // output
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype,
    const std::string_view& unary_post_op,
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    std::string_view unary_post_op_algorithm) {
  // todo W8A8 matmul kernel
  auto act_fp32 = (mat1.to(c10::kFloat) - input_zero_point) * input_scale;
  auto weight_fp32 =
      (mat2.to(c10::kFloat) - weight_zero_points) * weight_scales;
  // std::cout << "act_fp32" << act_fp32.sizes() << "weight_fp32" <<
  // weight_fp32.sizes() << std::endl;
  result = at::matmul(act_fp32, weight_fp32);
  if (bias.has_value()) {
    result += bias.value();
  }
  if (unary_post_op == "relu") {
    result = at::relu(result);
  } else if (unary_post_op == "tanh") {
    result = at::tanh(result);
  } else if (unary_post_op == "gelu") {
    result = at::gelu(result);
  }
  int64_t clamp_min, clamp_max;
  switch (output_dtype) {
    case c10::ScalarType::Byte:
      clamp_min = 0;
      clamp_max = 255;
      break;
    case c10::ScalarType::Char:
      clamp_min = -128;
      clamp_max = 127;
      break;
    default:
      result = result.to(output_dtype);
      return;
  }
  result = (result / output_scale + output_zero_point)
               .round()
               .clamp(clamp_min, clamp_max)
               .to(output_dtype);
  return;
}

static at::Tensor q_linear_per_tensor(
    at::Tensor act, // act
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight, // weight
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view post_op_name,
    torch::List<std::optional<at::Scalar>> post_op_args,
    std::string_view post_op_algorithm,
    std::string_view flag) {
  c10::musa::MUSAGuard device_guard(act.device());
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();
  const int64_t dim = act.dim();
  TORCH_CHECK(dim == 2, "qlinear musa: input dim should be 2, but got", dim);
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M,K] * [K, N]
  int64_t N = weight.size(1);
  std::cout << "M, N, K" << M << K << N << std::endl;

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};
  auto dst_dtype = output_dtype.value_or(act.scalar_type());

  Tensor qout = at::empty(dst_dims, act.options().dtype(dst_dtype));
  if (flag == "kernel") {
    at::native::matmul_stub(
        kMUSA,
        M,
        N,
        K,
        act,
        act_scale,
        act_zero_point,
        weight,
        weight_scales,
        weight_zero_points,
        b_raw,
        qout,
        output_scale,
        output_zero_point,
        dst_dtype,
        post_op_name);
  } else {
    quantized_matmul_per_channel(
        act.contiguous(),
        act_scale,
        act_zero_point,
        weight.contiguous(),
        weight_scales,
        weight_zero_points,
        bias,
        qout,
        output_scale,
        output_zero_point,
        dst_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm);
  }
  return qout;
}
} // namespace musa
} // namespace at
int main() {
  int64_t q_min = 0, q_max = 255;
  int64_t w_min = -128, w_max = 127;
  int64_t batch_size = 10, in_features = 7, out_features = 10;

  at::Tensor act = torch::randint(
      q_min, q_max + 1, {batch_size, in_features}, torch::kByte); //[)
  at::Tensor weight = torch::randint(
      w_min, w_max + 1, {in_features, out_features}, torch::kInt8);
  double act_scale = 1.0;
  int64_t act_zero_point = 127;

  at::Tensor weight_scales = torch::ones({out_features}, torch::kFloat32);
  at::Tensor weight_zero_points = torch::zeros({out_features}, torch::kInt64);
  // for (int64_t i = 0; i < out_features; ++i) {
  //     // 假设每个通道的原始浮点范围在 [-1.0, 1.0] 到 [-0.1, 0.1] 之间随机
  //     double channel_float_range = torch::rand({}).item<double>() * (1.0 -
  //     0.1) + 0.1; // Range between 0.1 and 1.0 weight_scales[i] =
  //     channel_float_range / (q_max - q_min); // Scale for this channel
  // }
  auto bias = torch::zeros({out_features}, torch::kFloat);
  auto bias1 = torch::rand({out_features}, c10::kFloat).mul_(10.0).sub_(5.0);
  double output_scale = 1.0;
  int64_t output_zero_point = 0.0;

  std::optional<c10::ScalarType> output_dtype = c10::kFloat;
  std::string_view post_op_name = "";
  torch::List<std::optional<at::Scalar>> post_op_args;
  std::string_view post_op_algorithm = "";
  at::Device device(at::kMUSA);
  auto start = std::chrono::high_resolution_clock::now();
  auto output1 = at::musa::q_linear_per_tensor(
      act.to(device),
      act_scale,
      act_zero_point,
      weight.to(device),
      weight_scales.to(device),
      weight_zero_points.to(device),
      bias.to(device),
      output_scale,
      output_zero_point,
      output_dtype,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      "ori");
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "output" << output1;
  std::cout << "duration" << duration.count() / 1000000.0 << "s" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto output2 = at::musa::q_linear_per_tensor(
      act.to(device),
      act_scale,
      act_zero_point,
      weight.to(device),
      weight_scales.to(device),
      weight_zero_points.to(device),
      bias.to(device),
      output_scale,
      output_zero_point,
      output_dtype,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      "kernel");
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "duration" << duration.count() / 1000000.0 << "s" << std::endl;
  // std::cout << "output" << output2 << std::endl;
  bool are_close =
      torch::allclose(output1.to("cpu"), output2.to("cpu"), 1e-3, 1e-3);
  std::cout << "are_close" << are_close << std::endl;

  auto output3 = at::musa::q_linear_per_tensor(
      act.to(device),
      act_scale,
      act_zero_point,
      weight.to(device),
      weight_scales.to(device),
      weight_zero_points.to(device),
      bias1.to(device),
      output_scale,
      output_zero_point,
      output_dtype,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      "ori");

  auto output4 = at::musa::q_linear_per_tensor(
      act.to(device),
      act_scale,
      act_zero_point,
      weight.to(device),
      weight_scales.to(device),
      weight_zero_points.to(device),
      bias1.to(device),
      output_scale,
      output_zero_point,
      output_dtype,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      "kernel");

  // std::cout << "output" << output3 << std::endl;
  // std::cout << "output" << output4 << std::endl;
  are_close = torch::allclose(output3.to("cpu"), output4.to("cpu"), 1e-3, 1e-3);
  std::cout << "are_close" << are_close << std::endl;
}
