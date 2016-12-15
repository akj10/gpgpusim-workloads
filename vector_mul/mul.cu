#include <stdio.h>
unsigned int N = 1 << 12;
unsigned int N_p = N/4;

__global__
void mul(int n, int *x, int *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * y[i];
}

int main(void)
{
  int *d_x, *d_y;
  int *x, *y;
  
  x = (int*)malloc(N*sizeof(int));
  y = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_x, N*sizeof(int)); 
  cudaMalloc(&d_y, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    x[i] = i%16;
    y[i] = i%16;
  }

  cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Perform SAXPY on 1M elements
  mul<<<(N+255)/256, 256>>>(N, d_x, d_y);

  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();


  int maxError = 0.0;
  for (int i = 0; i < N; i++) {
    //maxError = max(maxError, y[i]-(i+i));
    if (y[i] != ((i%16)*(i%16)))
      printf("Elements at pos %d not matching: y[i]=%x, expected = %x, i*i=%x\n", i,(int8_t)y[i], (int8_t)(x[i]*x[i]), ((i%16)*(i%16)));
  }
  printf("Max error: %d\n", maxError);
}
