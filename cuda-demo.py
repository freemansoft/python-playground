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

numIterations = 10000000
atype = np.float64
# warm up the jit
WARM_ITER = 5


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
# @numba.cuda.jit cuda flags
# device=True                    Installed on device.  callable from the device. 
# device=False (default, Global) Installed on device.  Only callable from host

# standard python speed
def math_cpu_nojit(param_triple):
   return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# jit gives 20x performance
@jit
def math_cpu_jit(param_triple):
   return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# cuda 'device' functions can return values
@cuda.jit('float64(float64[:])', device=True)
def math_gpu_jit(param_triple):
   return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

def calc_cpu_nojit_serial(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in range(iteration_count):
        param_array = all_params[i]
        result[i] = math_cpu_jit(param_array)

# normal function to run on cpu
# Note: JIT changes the way divide by zero is exposed - returns error set
@jit(parallel=False)
def calc_cpu_jit_serial(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in prange(iteration_count):
        param_array = all_params[i]
        result[i] = math_cpu_jit(param_array)

# normal function to run on cpu
# Note: JIT changes the way divide by zero is exposed - returns error set
@jit(parallel=True)
def calc_cpu_jit_parallel(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in prange(iteration_count):
        param_array = all_params[i]
        result[i] = math_cpu_jit(param_array)

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
        result[index] = math_gpu_jit(param_array)

# sums a vector when called from the CPU
@cuda.reduce
def reduce_on_gpu(a,b):
    return a+b

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

# exist to have same pattern as GPU. 
def driver_cpu_nojit_serial(iteration_count, all_params):
    result = np.zeros(iteration_count, dtype=atype)
    calc_cpu_nojit_serial(result,iteration_count,all_params)
    return result

# exist to have same pattern as GPU. 
def driver_cpu_jit_serial(iteration_count, all_params):
    result = np.zeros(iteration_count, dtype=atype)
    calc_cpu_jit_serial(result,iteration_count,all_params)
    return result

# exist to have same pattern as GPU. 
def driver_cpu_jit_parallel(iteration_count, all_params):
    result = np.zeros(iteration_count, dtype=atype)
    calc_cpu_jit_parallel(result,iteration_count,all_params)
    return result

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
    #if iteration_count > WARM_ITER:
    #    reduced_result = reduce_on_gpu(result_device)
    #     print("                 gpu calculated result :",reduced_result)
    return result

#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

def do_run(iterations, the_func, label, client=None):
    # create a one long 1D array with synthetic test data
    # create a 2D numIterations long array of 1x3 arrays so [numIterations[1,3]]
    one_dimension = np.arange(iterations, dtype=atype)
    one_dimension[one_dimension>=0]+=1.0
    triplets = np.stack((one_dimension,one_dimension,one_dimension),-1)

    start = timer()
    result = the_func(iterations, triplets)
    #uncomment this line to see parallel execution optimization
    #the_func.parallel_diagnostics(level=2)
    
    # print with sum to see if we get same answers on each track
    span = timer() - start
    if iterations >= 10:
        print(label, f"{span:,.4f}"," Iterations:",f"{iterations:,}", " dtype=", atype.__name__, "numcalc:", "sum(",f"{len(result):,}","):", np.sum(result))


if __name__=="__main__": 
    
    print("")
    do_run(WARM_ITER,     driver_gpu,              "GPU JIT warmup:   ")
    do_run(numIterations, driver_gpu,              "GPU grid:         ")
    # run cpu twice - jit overhead on first run
    do_run(WARM_ITER,     driver_cpu_jit_serial,   "CPU: JIT warmup   ")
    do_run(WARM_ITER,     driver_cpu_jit_parallel, "CPU: JIT warmup   ")
    do_run(numIterations, driver_cpu_jit_serial,   "CPU jit serial:   ")
    do_run(numIterations, driver_cpu_jit_parallel, "CPU jit parallel: ")

    do_run(numIterations, driver_cpu_nojit_serial, "CPU serial nojit: ")
 
