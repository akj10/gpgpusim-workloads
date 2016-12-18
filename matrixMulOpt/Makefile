
CC = nvcc
CCFLAGS = -arch=sm_20 #--ptxas-options=-v
CCDEBUGFLAGS = -g
#INC = -I./include -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc
default:
	$(CC) $(CCFLAGS) matrixMul.cu -o matrixMul $(INC) $(CCOPTIONS)
	$(CC) $(CCFLAGS) --ptx matrixMul.cu -o matrixMul.ptx $(INC) $(CCOPTIONS)
packed:
	$(CC) $(CCFLAGS) matrixMul_packed.cu -o matrixMul_packed $(INC) $(CCOPTIONS)
	$(CC) $(CCFLAGS) --ptx matrixMul_packed.cu -o mmp.ptx $(INC) $(CCOPTIONS)
int8:
	$(CC) $(CCFLAGS) matrixMul8.cu -o mm8 $(INC) $(CCOPTIONS)
	$(CC) $(CCFLAGS) --ptx matrixMul8.cu -o mm8.ptx $(INC) $(CCOPTIONS)
opt:
	$(CC) $(CCFLAGS) matrixMul_packed_opt.cu -o mmp_opt $(INC) $(CCOPTIONS)
	$(CC) $(CCFLAGS) --ptx matrixMul_packed_opt.cu -o mmp_opt.ptx $(INC) $(CCOPTIONS)
ptx:
	$(CC) $(CCFLAGS) --ptx matrixMul.cu -o test.ptx $(INC) $(CCOPTIONS)
ptx_packed:
	$(CC) $(CCFLAGS) --ptx matrixMul_packed.cu -o test.ptx $(INC) $(CCOPTIONS)
run:
	./matrixMul
run_packed:
	./matrixMul_packed
clean:
	rm -f matrixMul matrixMul_packed  test.ptx mm8 
	rm -f _cuobjdump* 
	rm -f gpgpusim_power_report*
