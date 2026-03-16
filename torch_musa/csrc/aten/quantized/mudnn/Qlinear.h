#ifndef Q_H
#define Q_H

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {
using matmul_kernel_fn = void (*)(
    int,
    int,
    int,
    at::Tensor,
    double,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor&,
    double,
    int64_t,
    c10::ScalarType,
    const std::string_view&);
DECLARE_DISPATCH(matmul_kernel_fn, matmul_stub);

} // namespace native
} // namespace at
#endif