#include <stdio.h>

__global__
void add(int n, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<10;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = i;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Perform SAXPY on 1M elements
  add<<<(N+255)/256, 256>>>(N, d_x, d_y);

  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-(i+i)));
    if (y[i] != 2.0*i) printf("Elements at pos %d not matching: y[i]=%f, 2*i=%f\n", i, y[i], 2.0*i);
  }
  printf("Max error: %f\n", maxError);
}
