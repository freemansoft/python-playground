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

atype = np.float64

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
# @numba.cuda.jit cuda flags
# device=True                    Installed on device.  callable from the device. 
# device=False (default, Global) Installed on device.  Only callable from host

# standard python speed
def math_nojit(param_triple):
    return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# jit gives 20x performance on CPU
# numba knows to compile this for both CUDA and CPU based on how the caller is compiled
@jit
def math_jit(param_triple):
    return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

# normal function to run on cpu
def calc_cpu_nojit_serial(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in range(iteration_count):
        param_array = all_params[i]
        result[i] = math_jit(param_array)

# normal function to run on cpu
# prange will act like range  loop if jit has parallel disabled
# Note: JIT changes the way divide by zero is exposed - returns error set
@jit(parallel=False)
def calc_cpu_jit_serial(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in prange(iteration_count):
        param_array = all_params[i]
        result[i] = math_jit(param_array)

# normal function to run on cpu
# prange will parallelize loop if jit has parallel enabled
# Note: JIT changes the way divide by zero is exposed - returns error set
@jit(parallel=True)
def calc_cpu_jit_parallel(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in prange(iteration_count):
        param_array = all_params[i]
        result[i] = math_jit(param_array)

# -------------------------------------------------------------------------------
# drive adapter for each run type
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

# -------------------------------------------------------------------------------
# common driver
# -------------------------------------------------------------------------------

# Invoke dask with different clients if client supplied
# Invoke cpu and gpu if client not provided
def do_run(iterations, the_func, label, client=None):
    # create a one long 1D array with synthetic test data
    # create a 2D numIterations long array of 1x3 arrays so [numIterations[1,3]]
    one_dimension = np.arange(iterations, dtype=atype)
    one_dimension[one_dimension>=0]+=1.0
    triplets = np.stack((one_dimension,one_dimension,one_dimension),-1)

    start = timer()
    if client is None:
        result = the_func(iterations, triplets)
        #uncomment this line to see parallel execution optimization
        #the_func.parallel_diagnostics(level=2)
    else :
        result = the_func(client, iterations,triplets)    
    # print with sum to see if we get same answers on each track
    span = timer() - start
    if iterations >= 10:
        print(label, f"{span:,.4f}"," Iterations:",f"{iterations:,}", " dtype=", atype.__name__, "numcalc:", "sum(",f"{len(result):,}","):", np.sum(result))
