.PHONY: all clean

all: microgpt_c microgpt_multicore microgpt_opencl microgpt_cuda

microgpt_c: microgpt.fut
	futhark c $< -o $@

microgpt_multicore: microgpt.fut
	futhark multicore $< -o $@

microgpt_opencl: microgpt.fut
	futhark opencl $< -o $@

microgpt_cuda: microgpt.fut
	futhark cuda $< -o $@

clean:
	rm -f microgpt_c microgpt_multicore microgpt_opencl microgpt_cuda \
		microgpt_c.c microgpt_multicore.c microgpt_opencl.c microgpt_cuda.c
