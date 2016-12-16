
CC = nvcc
CCFLAGS = -arch=sm_20 #--ptxas-options=-v
CCDEBUGFLAGS = -g
#INC = -I./include -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc
default:
	$(CC) $(CCFLAGS) matrixMul.cu -o matrixMul $(INC) $(CCOPTIONS)
packed:
	$(CC) $(CCFLAGS) matrixMul_packed.cu -o matrixMul_packed $(INC) $(CCOPTIONS)
ptx:
	$(CC) $(CCFLAGS) --ptx matrixMul.cu -o test.ptx $(INC) $(CCOPTIONS)
ptx_packed:
	$(CC) $(CCFLAGS) --ptx matrixMul_packed.cu -o test.ptx $(INC) $(CCOPTIONS)
run:
	./matrixMul
run_packed:
	./matrixMul_packed
clean:
	rm -f matrixMul matrixMul_packed  test.ptx
	rm -f _cuobjdump* 
	rm -f gpgpusim_power_report*
