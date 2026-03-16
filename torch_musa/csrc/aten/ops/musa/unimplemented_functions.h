#ifndef TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
#define TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H

#include <musa_runtime_api.h>
#include <musparse.h>
#include <memory>

typedef enum {
  MUSPARSE_STATUS_ALLOC_FAILED = 2,
  MUSPARSE_STATUS_MAPPING_ERROR = 5,
  MUSPARSE_STATUS_EXECUTION_FAILED = 12,
  MUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 13,
  MUSPARSE_STATUS_NOT_SUPPORTED = 10,
  MUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
} musparseStatus_t_new;

#define MUSPARSE_SOLVE_POLICY_NO_LEVEL static_cast<musparseSolvePolicy_t>(0)
#define MUSPARSE_SPGEMM_DEFAULT static_cast<musparseSpGEMMAlg_t>(0)
#define cusparseCcsrmm2 musparseCcsrmm
#define cusparseDcsrmm2 musparseDcsrmm
#define cusparseScsrmm2 musparseScsrmm
#define cusparseZcsrmm2 musparseZcsrmm

namespace c10 {
namespace musa {

__host__ musaError_t
musaThreadExchangeStreamCaptureMode(enum musaStreamCaptureMode* mode);

__host__ musaError_t musaStreamGetCaptureInfo(
    musaStream_t stream,
    enum musaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out);

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 4020)
__host__ musaError_t
musaGraphDebugDotPrint(musaGraph_t graph, const char* path, unsigned int flags);
#endif

__host__ musaError_t
musaMemPoolTrimTo(musaMemPool_t memPool, size_t minBytesToKeep);

__host__ musaError_t
musaMallocAsync(void** devPtr, size_t size, musaStream_t hStream);

__host__ musaError_t musaFreeAsync(void* devPtr, musaStream_t hStream);

__host__ musaError_t
musaDeviceGetDefaultMemPool(musaMemPool_t* memPool, int device);

__host__ musaError_t musaMemPoolGetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

__host__ musaError_t musaMemPoolSetAttribute(
    musaMemPool_t memPool,
    enum musaMemPoolAttr attr,
    void* value);

} // namespace musa
} // namespace c10

namespace at {
class Tensor;

namespace musa {
namespace sparse {
using musparseMatDescr = std::remove_pointer_t<musparseMatDescr_t>;
using musparseDnMatDescr = std::remove_pointer_t<musparseDnMatDescr_t>;
using musparseDnVecDescr = std::remove_pointer_t<musparseDnVecDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<musparseSpMatDescr_t>;
using musparseSpMatDescr = std::remove_pointer_t<musparseSpMatDescr_t>;
struct musparseSpGEMMDescr;
typedef struct musparseSpGEMMDescr* musparseSpGEMMDescr_t;
struct bsrsv2Info;
struct bsrsm2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;
typedef struct bsrsm2Info* bsrsm2Info_t;
musparseStatus_t musparseCreateBsrsv2Info(bsrsv2Info_t* info);
musparseStatus_t musparseDestroyBsrsv2Info(bsrsv2Info_t info);
musparseStatus_t musparseCreateBsrsm2Info(bsrsm2Info_t* info);
musparseStatus_t musparseDestroyBsrsm2Info(bsrsm2Info_t info);
musparseStatus_t musparseSpGEMM_destroyDescr(musparseSpGEMMDescr_t descr);
musparseStatus_t musparseSpGEMM_createDescr(musparseSpGEMMDescr_t* descr);

template <typename T, musparseStatus_t (*destructor)(T*)>
struct CuSparseDescriptorDeleterForMUSA {
  void operator()(T* x) {
    if (x != nullptr) {
      destructor(x);
    }
  }
};

template <typename T, musparseStatus_t (*destructor)(T*)>
class CuSparseDescriptorForMUSA {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuSparseDescriptorDeleterForMUSA<T, destructor>>
      descriptor_;
};

class CuSparseSpGEMMDescriptor : public CuSparseDescriptorForMUSA<
                                     musparseSpGEMMDescr,
                                     &musparseSpGEMM_destroyDescr> {
 public:
  CuSparseSpGEMMDescriptor() {
    musparseSpGEMMDescr_t raw_descriptor = nullptr;
    musparseSpGEMM_createDescr(&raw_descriptor);
    descriptor_.reset(raw_descriptor);
  }
};

class CuSparseSpMatDescriptor : public CuSparseDescriptorForMUSA<
                                    musparseSpMatDescr,
                                    &musparseDestroySpMat> {};

class CuSparseSpMatCsrDescriptor : public CuSparseSpMatDescriptor {
 public:
  explicit CuSparseSpMatCsrDescriptor(const Tensor& input) {}

  std::tuple<int64_t, int64_t, int64_t> get_size() {
    int64_t rows = 0, cols = 0, nnz = 0;
    musparseSpMatGetSize(this->descriptor(), &rows, &cols, &nnz);
    return std::make_tuple(rows, cols, nnz);
  }
  void set_tensor(const Tensor& input) {
    /*auto crow_indices = input.crow_indices();
    auto col_indices = input.col_indices();
    auto values = input.values();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
    TORCH_MUSASPARSE_CHECK(musparseCsrSetPointers(
        this->descriptor(),
        crow_indices.data_ptr(),
        col_indices.data_ptr(),
        values.data_ptr()));
    */
  }

  /*
   void set_mat_fill_mode(bool upper) {
     musparseFillMode_t fill_mode =
         upper ? MUSPARSE_FILL_MODE_UPPER : MUSPARSE_FILL_MODE_LOWER;
     TORCH_MUSASPARSE_CHECK(musparseSpMatSetAttribute(
         this->descriptor(),
         MUSPARSE_SPMAT_FILL_MODE,
         &fill_mode,
         sizeof(fill_mode)));
   }

   void set_mat_diag_type(bool unit) {
     musparseDiagType_t diag_type =
         unit ? MUSPARSE_DIAG_TYPE_UNIT : MUSPARSE_DIAG_TYPE_NON_UNIT;
     TORCH_MUSASPARSE_CHECK(musparseSpMatSetAttribute(
         this->descriptor(),
         MUSPARSE_SPMAT_DIAG_TYPE,
         &diag_type,
         sizeof(diag_type)));
   }
 */
};

musparseStatus_t musparseSpGEMM_workEstimation(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    musparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize1,
    void* externalBuffer1);

musparseStatus_t musparseCcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    muComplex* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer);
musparseStatus_t musparseDcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const double* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    double* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer);
musparseStatus_t musparseScsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const float* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    float* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer);
musparseStatus_t musparseZcsrgeam2(
    musparseHandle_t handle,
    int m,
    int n,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    muDoubleComplex* csrSortedValC,
    int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    void* pBuffer);

musparseStatus_t musparseCcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const muComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes);
musparseStatus_t musparseDcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const double* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const double* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes);
musparseStatus_t musparseScsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const float* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const float* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes);
musparseStatus_t musparseZcsrgeam2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    const muDoubleComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes);

musparseStatus_t musparseDbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseCbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseSbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseZbsrsv2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);

musparseStatus_t musparseCbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseDbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseSbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseZbsrsv2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const musparseMatDescr_t descrA,
    muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    int* pBufferSizeInBytes);

musparseStatus_t musparseCbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const muComplex* B,
    int ldb,
    muComplex* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseDbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const double* alpha,
    const musparseMatDescr_t descrA,
    const double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const double* B,
    int ldb,
    double* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseSbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const float* alpha,
    const musparseMatDescr_t descrA,
    const float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const float* B,
    int ldb,
    float* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseZbsrsm2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    const muDoubleComplex* B,
    int ldb,
    muDoubleComplex* X,
    int ldx,
    musparseSolvePolicy_t policy,
    void* pBuffer);

musparseStatus_t musparseSbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const float* alpha,
    const musparseMatDescr_t descrA,
    const float* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const float* f,
    float* x,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseCbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const muComplex* f,
    muComplex* x,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseDbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const double* alpha,
    const musparseMatDescr_t descrA,
    const double* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const double* f,
    double* x,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseZbsrsv2_solve(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    int mb,
    int nnzb,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedValA,
    const int* bsrSortedRowPtrA,
    const int* bsrSortedColIndA,
    int blockDim,
    bsrsv2Info_t info,
    const muDoubleComplex* f,
    muDoubleComplex* x,
    musparseSolvePolicy_t policy,
    void* pBuffer);

musparseStatus_t musparseCbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseDbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseSbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes);
musparseStatus_t musparseZbsrsm2_bufferSize(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    int* pBufferSizeInBytes);

musparseStatus_t musparseCbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseDbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const double* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseSbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const float* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);
musparseStatus_t musparseZbsrsm2_analysis(
    musparseHandle_t handle,
    musparseDirection_t dirA,
    musparseOperation_t transA,
    musparseOperation_t transXY,
    int mb,
    int n,
    int nnzb,
    const musparseMatDescr_t descrA,
    const muDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr,
    const int* bsrSortedColInd,
    int blockSize,
    bsrsm2Info_t info,
    musparseSolvePolicy_t policy,
    void* pBuffer);

musparseStatus_t musparseXcsrgeam2Nnz(
    musparseHandle_t handle,
    int m,
    int n,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrC,
    int* csrSortedRowPtrC,
    int* nnzTotalDevHostPtr,
    void* workspace);

} // namespace sparse
} // namespace musa
namespace native {

typedef enum {
  MUSPARSE_SPMM_ALG_DEFAULT = 0,
  MUSPARSE_SPMM_COO_ALG1 = 1,
  MUSPARSE_SPMM_COO_ALG2 = 2,
  MUSPARSE_SPMM_COO_ALG3 = 3,
  MUSPARSE_SPMM_COO_ALG4 = 5,
  MUSPARSE_SPMM_CSR_ALG1 = 4,
  MUSPARSE_SPMM_CSR_ALG2 = 6,
  MUSPARSE_SPMM_CSR_ALG3 = 12,
  MUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13,
  MUSPARSE_SPMM_BSR_ALG1 = 14
} musparseSpMMAlg_t;

typedef enum {
  MUSPARSE_ORDER_COL = 1, ///< Column-Major Order - Matrix memory layout
  MUSPARSE_ORDER_ROW = 2 ///< Row-Major Order - Matrix memory layout
} musparseOrder_t;

typedef void* csrgemm2Info_t;
musparseStatus_t musparseCreateCsrgemm2Info(csrgemm2Info_t* info);
musparseStatus_t musparseDestroyCsrgemm2Info(csrgemm2Info_t info);

musparseStatus_t musparseCreateDnMat(
    musparseDnMatDescr_t* dnMatDescr,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    void* values,
    musaDataType valueType,
    musparseOrder_t order);

musparseStatus_t musparseSpMM(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseDnMatDescr_t matB,
    const void* beta,
    musparseDnMatDescr_t matC,
    musaDataType computeType,
    musparseSpMMAlg_t alg,
    void* externalBuffer);
musparseStatus_t musparseSpMM_bufferSize(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseDnMatDescr_t matB,
    const void* beta,
    musparseDnMatDescr_t matC,
    musaDataType computeType,
    musparseSpMMAlg_t alg,
    size_t* bufferSize);

musparseStatus_t musparseXcsrgemm2Nnz(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    int* csrSortedRowPtrC,
    int* nnzTotalDevHostPtr,
    const csrgemm2Info_t info,
    void* pBuffer);

musparseStatus_t musparseCcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const muComplex* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    muComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer);

musparseStatus_t musparseDcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const double* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const double* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const double* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const double* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    double* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer);

musparseStatus_t musparseScsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const float* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const float* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const float* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const float* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    float* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer);

musparseStatus_t musparseZcsrgemm2(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const muDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const muDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const muDoubleComplex* csrSortedValD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    const musparseMatDescr_t descrC,
    muDoubleComplex* csrSortedValC,
    const int* csrSortedRowPtrC,
    int* csrSortedColIndC,
    const csrgemm2Info_t info,
    void* pBuffer);

musparseStatus_t musparseCcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes);

musparseStatus_t musparseDcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const double* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const double* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes);

musparseStatus_t musparseScsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const float* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const float* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes);

musparseStatus_t musparseZcsrgemm2_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int k,
    const muDoubleComplex* alpha,
    const musparseMatDescr_t descrA,
    int nnzA,
    const int* csrSortedRowPtrA,
    const int* csrSortedColIndA,
    const musparseMatDescr_t descrB,
    int nnzB,
    const int* csrSortedRowPtrB,
    const int* csrSortedColIndB,
    const muDoubleComplex* beta,
    const musparseMatDescr_t descrD,
    int nnzD,
    const int* csrSortedRowPtrD,
    const int* csrSortedColIndD,
    csrgemm2Info_t info,
    size_t* pBufferSizeInBytes);

namespace sparse {
namespace musa {
musparseStatus_t musparseXcsrsort_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int* csrRowPtrA,
    const int* csrColIndA,
    size_t* pBufferSizeInBytes);
musparseStatus_t musparseXcoosort_bufferSizeExt(
    musparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int* cooRowsA,
    const int* cooColsA,
    size_t* pBufferSizeInBytes);
} // namespace musa
namespace impl::musa {

musparseStatus_t musparseXbsrsv2_zeroPivot(
    musparseHandle_t handle,
    at::musa::sparse::bsrsv2Info_t info,
    int* position);
musparseStatus_t musparseXbsrsm2_zeroPivot(
    musparseHandle_t handle,
    at::musa::sparse::bsrsm2Info_t info,
    int* position);
musparseStatus_t musparseSpGEMM_compute(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    at::musa::sparse::musparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize2,
    void* externalBuffer2);
musparseStatus_t musparseSpGEMM_copy(
    musparseHandle_t handle,
    musparseOperation_t opA,
    musparseOperation_t opB,
    const void* alpha,
    musparseSpMatDescr_t matA,
    musparseSpMatDescr_t matB,
    const void* beta,
    musparseSpMatDescr_t matC,
    musaDataType computeType,
    musparseSpGEMMAlg_t alg,
    at::musa::sparse::musparseSpGEMMDescr_t spgemmDescr);

} // namespace impl::musa
} // namespace sparse
} // namespace native
} // namespace at

namespace at::native::sparse {
using SparseTensor = at::Tensor;
} // namespace at::native::sparse

#endif // TORCH_MUSA_CSRC_ATEN_OPS_MUSA_UNIMPLEMENTED_FUNCTIONS_H
