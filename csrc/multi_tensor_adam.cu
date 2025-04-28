#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum{
  ADAM_MODE_0   =0, // L2 regularization mode
  ADAM_MODE_1   =1  // Decoupled weight decay mode(AdamW)
} adamMode_t;

using MATH_T = float;

template<typename T, typename FULL_T, typename index_t>
struct AdamFunctor
{
   __device__ __forceinline__ void operator()(
    index_t chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    const float lr,
    adamMode_t mode,
    const float decay)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(index_t i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
          r_v[ii] = v[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
          m[i] = r_m[ii];
          v[i] = r_v[ii];
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct AdamCapturableFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    adamMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          m[i] = static_cast<T>(r_m[ii]);
          v[i] = static_cast<T>(r_v[ii]);
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct AdamCapturableMasterFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    adamMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    FULL_T* p_master = (FULL_T*)tl.addresses[4][tensor_loc];
    p_master += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          m[i] = static_cast<FULL_T>(r_m[ii]);
          v[i] = static_cast<FULL_T>(r_v[ii]);
        }
      }
    }
  }
};

void multi_tensor_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay)
{
  using namespace at;

  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  size_t max_size = 0;
  bool requires_64bit_indexing = false;
  for (auto it = tensor_lists.begin(); it != tensor_lists.end(); it++) {
    for (auto it2 = it->begin(); it2 != it->end(); it2++) {
      if (it2->numel() > max_size) {
        max_size = it2->numel();
	if (max_size >= INT_MAX) {
          requires_64bit_indexing = true;
	  break;
        }
      }
    }
    if (requires_64bit_indexing) {
      break;
    }
  }

  if (requires_64bit_indexing) {
    // Assume single type across p,g,m1,m2 now
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
      tensor_lists[0][0].scalar_type(), 0, "adam",
      multi_tensor_apply<4>(
        (int64_t) BLOCK_SIZE,
        (int64_t) chunk_size,
        noop_flag,
        tensor_lists,
        AdamFunctor<scalar_t_0, float, int64_t>(),
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        epsilon,
        lr,
        (adamMode_t) mode,
        weight_decay); )
  } else {
      // Assume single type across p,g,m1,m2 now
      DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "adam",
        multi_tensor_apply<4>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          AdamFunctor<scalar_t_0, float, int32_t>(),
          beta1,
          beta2,
          bias_correction1,
          bias_correction2,
          epsilon,
          lr,
          (adamMode_t) mode,
          weight_decay); )
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void multi_tensor_adam_capturable_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adam",
    multi_tensor_apply<4>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      AdamCapturableFunctor<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (adamMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

void multi_tensor_adam_capturable_master_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adam",
    multi_tensor_apply<5>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      AdamCapturableMasterFunctor<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (adamMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

template<typename T, typename FULL_T, typename index_t>
struct AdamFunctorNoUpdateMV
{
   __device__ __forceinline__ void operator()(
    index_t chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    const float lr,
    adamMode_t mode,
    const float decay)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(index_t i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
          r_v[ii] = v[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
          // m[i] = r_m[ii];
          // v[i] = r_v[ii];
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct AdamCapturableFunctorNoUpdateMV
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    adamMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          // m[i] = static_cast<T>(r_m[ii]);
          // v[i] = static_cast<T>(r_v[ii]);
        }
      }
    }
  }
};

template<typename T, typename FULL_T>
struct AdamCapturableMasterFunctorNoUpdateMV
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float beta1,
    const float beta2,
    const int* step,
    const int bias_correction,
    const float epsilon,
    const float* lr,
    adamMode_t mode,
    const float decay,
    const float* inv_scale)
  {
    if(*noop_gmem == 1)
      return;

    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (bias_correction == 1) {
      beta1_correction = 1 - pow(beta1, *step);
      beta2_correction = 1 - pow(beta2, *step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    FULL_T* p_master = (FULL_T*)tl.addresses[4][tensor_loc];
    p_master += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
      MATH_T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = static_cast<MATH_T>(g[i]) * (*inv_scale);
          g[i] = static_cast<T>(r_g[ii]);
          r_p[ii] = static_cast<MATH_T>(p_master[i]);
          r_m[ii] = static_cast<MATH_T>(m[i]);
          r_v[ii] = static_cast<MATH_T>(v[i]);
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if(mode == ADAM_MODE_0) { // L2
          r_g[ii] = r_g[ii] + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = next_m_unbiased / denom;
          r_p[ii] = r_p[ii] - (*lr * update);
        }
        else { // weight decay
          r_m[ii] = beta1 * r_m[ii] + (1-beta1) * r_g[ii];
          r_v[ii] = beta2 * r_v[ii] + (1-beta2) * r_g[ii] * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (*lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = static_cast<T>(r_p[ii]);
          p_master[i] = static_cast<FULL_T>(r_p[ii]);
          // m[i] = static_cast<FULL_T>(r_m[ii]);
          // v[i] = static_cast<FULL_T>(r_v[ii]);
        }
      }
    }
  }
};

void multi_tensor_adam_noupdate_mv_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay)
{
  using namespace at;

  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  size_t max_size = 0;
  bool requires_64bit_indexing = false;
  for (auto it = tensor_lists.begin(); it != tensor_lists.end(); it++) {
    for (auto it2 = it->begin(); it2 != it->end(); it2++) {
      if (it2->numel() > max_size) {
        max_size = it2->numel();
	if (max_size >= INT_MAX) {
          requires_64bit_indexing = true;
	  break;
        }
      }
    }
    if (requires_64bit_indexing) {
      break;
    }
  }

  if (requires_64bit_indexing) {
    // Assume single type across p,g,m1,m2 now
    DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
      tensor_lists[0][0].scalar_type(), 0, "adam",
      multi_tensor_apply<4>(
        (int64_t) BLOCK_SIZE,
        (int64_t) chunk_size,
        noop_flag,
        tensor_lists,
        AdamFunctorNoUpdateMV<scalar_t_0, float, int64_t>(),
        beta1,
        beta2,
        bias_correction1,
        bias_correction2,
        epsilon,
        lr,
        (adamMode_t) mode,
        weight_decay); )
  } else {
      // Assume single type across p,g,m1,m2 now
      DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "adam",
        multi_tensor_apply<4>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          AdamFunctorNoUpdateMV<scalar_t_0, float, int32_t>(),
          beta1,
          beta2,
          bias_correction1,
          bias_correction2,
          epsilon,
          lr,
          (adamMode_t) mode,
          weight_decay); )
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void multi_tensor_adam_noupdate_mv_capturable_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adam",
    multi_tensor_apply<4>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      AdamCapturableFunctorNoUpdateMV<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (adamMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

void multi_tensor_adam_noupdate_mv_capturable_master_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int mode,
  const int bias_correction,
  const float weight_decay,
  at::Tensor inv_scale)
{
  using namespace at;

  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adam",
    multi_tensor_apply<5>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      AdamCapturableMasterFunctorNoUpdateMV<scalar_t_0, float>(),
      beta1,
      beta2,
      step.data_ptr<int>(),
      bias_correction,
      epsilon,
      lr.data_ptr<float>(),
      (adamMode_t) mode,
      weight_decay,
      inv_scale.data_ptr<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}

template <typename T, typename FULL_T, typename index_t>
struct AdamWRollbackFunctor {
  __device__ __forceinline__ void operator()(
      index_t                chunk_size,
      volatile int*          noop_gmem,
      TensorListMetadata<4>& tl,
      const float            beta1,
      const float            beta2,
      const float            beta1_correction,   // pre-computed or 1.0
      const float            beta2_correction,   // pre-computed or 1.0
      const float            epsilon,
      const float            lr,
      adamMode_t             mode,
      const float            decay)
  {
    /* rollback supports AdamW (decoupled) only */
    assert(mode == ADAM_MODE_1 &&
           "AdamWRollbackFunctor supports ADAM_MODE_1 (AdamW) only");

    if (*noop_gmem == 1) return;

    /* ---- locate tensors ------------------------------------------------ */
    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];
    index_t chunk_idx  = tl.block_to_chunk[blockIdx.x];
    index_t n          = tl.sizes[tensor_loc];

    T*      g = (T*)     tl.addresses[0][tensor_loc] + chunk_idx * chunk_size;
    T*      p = (T*)     tl.addresses[1][tensor_loc] + chunk_idx * chunk_size;
    FULL_T* m = (FULL_T*)tl.addresses[2][tensor_loc] + chunk_idx * chunk_size;
    FULL_T* v = (FULL_T*)tl.addresses[3][tensor_loc] + chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    /* ---- main loop ----------------------------------------------------- */
    for (index_t i_start = 0;
         i_start < n && i_start < chunk_size;
         i_start += blockDim.x * ILP)
    {
      MATH_T r_g[ILP], r_p[ILP], r_m[ILP], r_v[ILP];

#pragma unroll
      for (int ii = 0; ii < ILP; ++ii) {
        index_t i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
          r_v[ii] = v[i];
        } else {
          r_g[ii] = r_p[ii] = r_m[ii] = r_v[ii] = MATH_T(0);
        }
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ++ii) {
        /* 1. Rebuild moments exactly as the *forward* AdamW step did */
        r_m[ii] = beta1 * r_m[ii] + (1.f - beta1) * r_g[ii];
        r_v[ii] = beta2 * r_v[ii] + (1.f - beta2) * r_g[ii] * r_g[ii];

        MATH_T m_hat = r_m[ii] / beta1_correction;
        MATH_T v_hat = r_v[ii] / beta2_correction;

        MATH_T denom  = sqrtf(v_hat) + epsilon;
        MATH_T update = m_hat / denom;            // weight-decay excluded

        /* 2. Undo Adam update */
        r_p[ii] += lr * update;

        /* 3. Undo weight-decay scaling (p_old = p_new / (1 - lr·decay)) */
        r_p[ii] /= (1.f - lr * decay);
      }

#pragma unroll
      for (int ii = 0; ii < ILP; ++ii) {
        index_t i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size)
          p[i] = static_cast<T>(r_p[ii]);   // ONLY p is restored
      }
    }
  }
};


void multi_tensor_adamw_rollback_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists, // g, p, m, v
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int   step,
  const int   mode,            // must be ADAM_MODE_1
  const int   bias_correction, // 0 or 1
  const float weight_decay)
{
using namespace at;

/* compute bias corrections only if requested */
float beta1_corr = 1.f, beta2_corr = 1.f;
if (bias_correction == 1) {
  beta1_corr = 1.f - std::pow(beta1, step);
  beta2_corr = 1.f - std::pow(beta2, step);
}

/* choose 32- vs 64-bit indexing ----------------------------------- */
bool use_64bit = false;
size_t max_sz = 0;
for (auto &grp : tensor_lists)
  for (auto &t : grp)
    if ((max_sz = std::max(max_sz, (size_t)t.numel())) >= INT_MAX) {
      use_64bit = true; break;
    }

if (use_64bit) {
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adamw_rollback",
    multi_tensor_apply<4>(
        (int64_t)BLOCK_SIZE,
        (int64_t)chunk_size,
        noop_flag,
        tensor_lists,
        AdamWRollbackFunctor<scalar_t_0, float, int64_t>(),
        beta1, beta2,
        beta1_corr, beta2_corr,
        epsilon, lr,
        static_cast<adamMode_t>(mode),
        weight_decay));
} else {
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adamw_rollback",
    multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        AdamWRollbackFunctor<scalar_t_0, float, int32_t>(),
        beta1, beta2,
        beta1_corr, beta2_corr,
        epsilon, lr,
        static_cast<adamMode_t>(mode),
        weight_decay));
}

AT_CUDA_CHECK(cudaGetLastError());
}
