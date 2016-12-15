#include <stdio.h>
unsigned int N = 1 << 12;
unsigned int N_p = N/4;

__global__
void add(unsigned int n, unsigned int *x, unsigned int *y)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + y[i];
}

int main(void)
{
  unsigned int /**x, *y,*/ *d_x, *d_y;
  int8_t *x, *y;

  x = (int8_t*)malloc(N*sizeof(int8_t));
  y = (int8_t*)malloc(N*sizeof(int8_t));

  cudaMalloc(&d_x, N_p*sizeof(unsigned int)); 
  cudaMalloc(&d_y, N_p*sizeof(unsigned int));

  for (unsigned int i = 0; i < N; i++) {
    x[i] = i%256;
    y[i] = i%256;
  }

  cudaMemcpy(d_x, (unsigned int*) x, N_p*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, (unsigned int*) y, N_p*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Perform SAXPY on 1M elements
  add<<<(N_p+255)/256, 256>>>(N_p, d_x, d_y);

  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, N*sizeof(int8_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  y = (int8_t*) y;

  int8_t maxError = 0;
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, (y[i]-(int8_t)(((i%256+i%256)%256))));
    if (y[i] != (int8_t)(2*(i%256))%256) printf("Elements at pos %d not matching: y[i]=%x, 2*i=%x\n", i, y[i], (2*(i%256))%256);
  }
  printf("Max error: %d\n", maxError);
}
