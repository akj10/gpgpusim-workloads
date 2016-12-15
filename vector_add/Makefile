CC = nvcc
CCFLAGS = -arch=sm_20 #--ptxas-options=-v
CCDEBUGFLAGS = -g
#INC = -I./include -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc
default:
	$(CC) $(CCFLAGS) add.cu -o add $(INC) $(CCOPTIONS)
packed:
	$(CC) $(CCFLAGS) add_packed.cu -o add_packed $(INC) $(CCOPTIONS)
ptx:
	$(CC) $(CCFLAGS) --ptx add.cu -o test.ptx $(INC) $(CCOPTIONS)
ptx_packed:
	$(CC) $(CCFLAGS) --ptx add_packed.cu -o test.ptx $(INC) $(CCOPTIONS)
run:
	./add
run_packed:
	./add_packed
clean:
	rm -f add add_packed  test.ptx
	rm -f _cuobjdump*
