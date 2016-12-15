#include <stdio.h>

#define N 32*2000
#define threadPerBlock 32
#define blockPerGrid min(2000 , (N+threadPerBlock-1) / threadPerBlock )
#define testBlocks blockPerGrid

double cpudot(int n, float *x, float *y)
{
  double z = 0.0f;
  for (int i=0; i<n; i++) z += x[i] * y[i];
  return z;
}

__global__
void dot(int n, float *x, float *y, float *z)
{
  __shared__ float cache[threadPerBlock];
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < n) {
    cache[threadIdx.x] = x[tid] * y[tid];
  }
  __syncthreads();
  //printf("Thread %d: x=%f, y=%f, cache=%f\n", tid, x[tid], y[tid], cache[threadIdx.x]);

  int i = threadPerBlock/2;
  while (i != 0) {
    if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    //printf("iter %d: tid=%d, cache=%f\n", i, threadIdx.x, cache[threadIdx.x]);
    i /= 2;
  }
  if (threadIdx.x == 0) z[blockIdx.x] = cache[0];
}

int main(void)
{
  float *x, *y, *z, *d_x, *d_y, *d_z;
  double cpu_result, gpu_result=0.0;
  double cpu_acc=0.0;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  z = (float*)malloc(blockPerGrid*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, blockPerGrid*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 0.5;
    y[i] = i;
    z[i%blockPerGrid] = 0;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, blockPerGrid*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Perform dot on 1M elements
  dot<<<blockPerGrid, threadPerBlock>>>(N, d_x, d_y, d_z);

  cudaDeviceSynchronize();
  cudaMemcpy(z, d_z, blockPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i=0; i<testBlocks; i++) {
    //printf("%f\n", z[i]);
    gpu_result += (double)z[i];
    cpu_result = cpudot(threadPerBlock, x+i*threadPerBlock, y+i*threadPerBlock);
    cpu_acc += cpu_result;
    if (abs(z[i] - cpu_result) > 0.000001) 
      printf("Block %d: Partial Dot Product not matching GPU:%f CPU:%f\n", i, z[i], cpu_result);
    else
      printf("Block %d: Partial Dot Product matches GPU:%f CPU:%f\n", i, z[i], cpu_result);
  }
  printf("cpu accumulated val=%f\n", cpu_acc);

  cpu_result = cpudot(testBlocks*threadPerBlock, x, y);
    if (abs(gpu_result - cpu_result) > 0.000001) printf("GPU Dot product:%f not matching with CPU:%f\n", gpu_result, cpu_result);
    else printf("GPU Dot product:%f matches with CPU:%f\n", gpu_result, cpu_result);
}
