CC = nvcc
CCFLAGS = -w -arch=sm_20 #--ptxas-options=-v
CCDEBUGFLAGS = -g
#INC = -I./include -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.5/samples/common/inc
default:
	$(CC) $(CCFLAGS) dot.cu -o dot $(INC) $(CCOPTIONS)
ptx:
	$(CC) $(CCFLAGS) --ptx dot.cu $(INC) $(CCOPTIONS)
run:
	./dot
clean:
	rm -f dot
