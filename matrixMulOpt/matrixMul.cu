#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <cuda_runtime.h>

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

#define BLOCK_SIZE 16

void matrixMulCPU(float *A, float *B, float *C, int size){
	int i, j, k;
	float sum;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++){
			sum = 0;
			for(k = 0; k < size; k++){
				sum += A[i * size + k] * B[j * size + k];
			}
			C[i * size + j] = sum;
		}

	}
}

__global__ void matrixMulGPU(float *A, float *B, float *C, int width){
        // Block index
        int bx = blockIdx.x;
    	int by = blockIdx.y;

    	// Thread index
    	int tx = threadIdx.x;
    	int ty = threadIdx.y;

    	// Index of the first sub-matrix of A processed by the block
    	int aBegin = width * BLOCK_SIZE * by;

    	// Index of the last sub-matrix of A processed by the block
    	int aEnd   = aBegin + width - 1;

    	// Step size used to iterate through the sub-matrices of A
    	int aStep  = BLOCK_SIZE;

    	// Index of the first sub-matrix of B processed by the block
    	int bBegin = width * BLOCK_SIZE * bx;

    	// Step size used to iterate through the sub-matrices of B
    	int bStep  = BLOCK_SIZE;

    	// Csub is used to store the element of the block sub-matrix
    	// that is computed by the thread
    	float Csub = 0;
        if (bx == 0 && tx==0 && by==0 && ty==0) {
            printf("aBegin=%d, bBegin=%d\n", aBegin, bBegin);
            printf("aStep=%d, bStep=%d\n", aStep, bStep);
            printf("aEnd=%d\n", aEnd);
        }

    	// Loop over all the sub-matrices of A and B
    	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    	{

        	// Declaration of the shared memory array As used to
        	// store the sub-matrix of A
        	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        	// Declaration of the shared memory array Bs used to
        	// store the sub-matrix of B
        	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        	// Load the matrices from device memory
        	// to shared memory; each thread loads
        	// one element of each matrix
        	As[ty][tx] = A[a + width * ty + tx];
        	Bs[ty][tx] = B[b + width * ty + tx];
            if (bx == 0 && tx==0 && by==0 && ty==0) {
                printf("Loading A=%f, B=%f\n", As[ty][tx], Bs[ty][tx]);
            }

        	// Synchronize to make sure the matrices are loaded
        	__syncthreads();

        	// Multiply the two matrices together;
        	// each thread computes one element
        	// of the block sub-matrix
#pragma unroll

        	for (int k = 0; k < BLOCK_SIZE; ++k)
        	{
            		Csub += As[ty][k] * Bs[tx][k];
        	}

        	// Synchronize to make sure that the preceding
        	// computation is done before loading two new
        	// sub-matrices of A and B in the next iteration
        	__syncthreads();
    	}

    	// Write the block sub-matrix to device memory;
    	// each thread writes one element
    	int c = width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    	C[c + width * ty + tx] = Csub;

	

}

int main(int argc, char *argv[]){
  	int i;
  	float *A, *B, *C, *D;
  	float *A_dev, *B_dev, *C_dev;
  	double start_timer, end_timer;

	int width, MSIZE;

	if(argc < 2){
		printf("Error input options\n");
		exit(1);
	}

	width = atoi(argv[1]);
	MSIZE = width * width;

    	A = (float*)malloc(sizeof(float)*MSIZE);
     	cudaMalloc(&A_dev, MSIZE*sizeof(float));
    	B = (float*)malloc(sizeof(float)*MSIZE);
    	cudaMalloc(&B_dev, MSIZE*sizeof(float));
    	C = (float*)malloc(sizeof(float)*MSIZE);
    	cudaMalloc(&C_dev, MSIZE*sizeof(float));
    	D = (float*)malloc(sizeof(float)*MSIZE);
  
	srand(time(NULL));
  	// Init matrix
    	for(i = 0; i < MSIZE; i++){
      		A[i] = ((double) rand() / (RAND_MAX)) + 1;
      		B[i] = ((double) rand() / (RAND_MAX)) + 1;
      		C[i] = 0;
      		D[i] = 0;
    	}

    	cudaMemcpy(A_dev, A, MSIZE*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(B_dev, B, MSIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_dev, C, MSIZE*sizeof(float), cudaMemcpyHostToDevice);
  	cudaDeviceSynchronize();

   	/*thread blcok conf.*/ 
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  	dim3 grid(width/dimBlock.x, width/dimBlock.y);

  	start_timer = my_timer();

    	matrixMulGPU<<<grid, dimBlock>>>(A_dev, B_dev, C_dev, width);

  	cudaDeviceSynchronize();

  	end_timer = my_timer();
  	printf("The GPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

    	cudaMemcpy(C, C_dev, MSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  	cudaDeviceSynchronize();
 
	start_timer = my_timer();

    	matrixMulCPU(A, B, D, width);
 
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);

  	//Verification
  	printf("Verifying\n");
	int flag = 0;
    	for(i = 0; i < MSIZE; i++){
      		if(abs(C[i] - D[i]) > 1e-3){
        		printf("Error:%f, %f, %d\n", C[i], D[i], i);
			break;
      		}
		flag ++;
	}
        if(flag == MSIZE) printf("Verify Success!!\n");

	// memory free
    	free(A);
    	cudaFree(A_dev);
    	free(B);
    	cudaFree(B_dev);
    	free(C);
    	cudaFree(C_dev);
    	free(D);
	
  return 0;
}
