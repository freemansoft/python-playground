# math and set operations
import numpy as np
# used for GPU
#import numba
from numba import jit
from numba import prange
from numba import cuda
# to measure exec time 
from timeit import default_timer as timer
import random
from time import sleep
import math 
import CalcCore

numIterations = 10000000
# warm up the jit
WARM_ITER = 5


# operates on individual element
# assumes this was called as part of a matrix call
# cuda global function cannot return values
@cuda.jit('void(float64[:],int32,float64[:,:])')
def calc_on_gpu(result, iteration_count, all_params):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    index = tx + ty * bw 
    if index < iteration_count: 
        param_array = all_params[index]
        result[index] = CalcCore.math_jit(param_array)

# sums a vector when called from the CPU
# Still may cause large data movement
@cuda.reduce
def reduce_on_gpu(a,b):
    return a+b

# map data into device memory and invoke and copy back
def driver_gpu(iteration_count, all_params):
    # calculate matrix size
    # rtx 2060 super has 34 SMs at 64-P32, 32-FP64 cores each for 2176 processors
    # CUDA sizing https://docs.nvidia.com/cuda/cuda-occupancy-calculator/CUDA_Occupancy_Calculator.xls
    threadsperblock = 1024
    blockspergrid = iteration_count // threadsperblock + 1
    if iteration_count > WARM_ITER: print("                 blocks/grid:",blockspergrid," threads/block:",threadsperblock)

    # transfers
    #   to the device are asynchronous 
    #   from the device are synchronous
    all_params_device = cuda.to_device(all_params)
    # allocate the result array on the device
    result_device = cuda.device_array(iteration_count, dtype=np.float64)
    # this magically spreads this function with one element from each array across all GPU processors
    calc_on_gpu[blockspergrid,threadsperblock](result_device, iteration_count, all_params_device)
    # copy back is an expensive information probabl 1/3 time on my machine
    result = result_device.copy_to_host()
    # reduce on the GPU side (using cuda.reduce) only bring back sum - slightly different answer
    # if iteration_count > WARM_ITER:
    #     reduced_result = reduce_on_gpu(result_device)
    #     print("                 gpu calculated result :",reduced_result)
    return result

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------


if __name__=="__main__": 
    
    print("")
    CalcCore.do_run(WARM_ITER,     driver_gpu,              "GPU JIT warmup:   ")
    CalcCore.do_run(numIterations, driver_gpu,              "GPU grid:         ")
    # run cpu twice - jit overhead on first run
    CalcCore.do_run(WARM_ITER,     CalcCore.driver_cpu_jit_parallel, "CPU: JIT warmup   ")
    CalcCore.do_run(WARM_ITER,     CalcCore.driver_cpu_jit_serial,   "CPU: JIT warmup   ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_jit_parallel, "CPU jit parallel: ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_jit_serial,   "CPU jit serial:   ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_nojit_serial, "CPU serial nojit: ")
 
