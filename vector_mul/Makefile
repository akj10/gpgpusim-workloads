CC = nvcc
CCFLAGS = -arch=sm_20 #--ptxas-options=-v
CCDEBUGFLAGS = -g
#INC = -I./include -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc
default:
	$(CC) $(CCFLAGS) mul.cu -o mul $(INC) $(CCOPTIONS)
packed:
	$(CC) $(CCFLAGS) mul_packed.cu -o mul_packed $(INC) $(CCOPTIONS)
ptx:
	$(CC) $(CCFLAGS) --ptx mul.cu -o test.ptx $(INC) $(CCOPTIONS)
ptx_packed:
	$(CC) $(CCFLAGS) --ptx mul_packed.cu -o test.ptx $(INC) $(CCOPTIONS)
run:
	./mul
run_packed:
	./mul_packed
clean:
	rm -f mul mul_packed  test.ptx
	rm -f _cuobjdump* 
