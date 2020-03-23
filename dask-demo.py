# math and set operations
import numpy as np
# used for GPU
import numba
from numba import jit
from numba import prange
# DASK
from dask.distributed import Client, LocalCluster
import dask
import dask.array as da
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
#                     DASK
# -------------------------------------------------------------------------------

def calc_dask_worker_threads(client):
    # number of execution workers
    workers = client.cluster.workers
    worker_count = len(workers)
    # grab one of the workers and find out how many threads it has
    one_worker = list(workers.values())[0]
    worker_thread_count = one_worker.nthreads
    #size the number of DASKS tasks to match the worker thread count
    total_worker_count = worker_thread_count * worker_count
    return total_worker_count

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

# standard python speed
def math_cpu_nojit(param_triple):
   return math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

# jit gives 20x performance
@jit
def math_cpu_jit(param_triple):
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
# DASK driver for each node - one in each task
def dask_driver_node(all_params):
    iteration_count = all_params.shape[0]
    result = np.zeros(iteration_count, dtype=atype)
    calc_cpu_jit_serial(result,iteration_count,all_params)
    return result

# master dask adapter 
# a is an array of 1x3 arrays
def dask_driver(client, iteration_count, all_inputs):
    #result = np.zeros(iteration_count, dtype=atype)
    numDaskTasks = calc_dask_worker_threads(client)
    # split the list into segments one for each dask task - length may vary
    a_list = np.array_split(all_inputs,numDaskTasks)
    print("                             numDaskTask:", numDaskTasks, " each around:", a_list[0].shape[0])

    # scatter the lists (sequences?) of 1x3 arrays to the task nodes returns list of futures
    # returns list of futures
    scattered_future = client.scatter(a_list)
    # execute this function across the cluster against the scattered data
    calculated_future = client.map(dask_driver_node, scattered_future)
    # returns array of result arrays
    results_gathered = client.gather(calculated_future)
    # concatenate the arrays of result arrays into a single array
    results_unified = np.concatenate(results_gathered)
    return results_unified


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

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

if __name__=="__main__": 
    print("")
    print("")

    # open all of these with dashboards. Open and close in sequence so no conflicts
    # Can open all at once if disable dashboard:  dashboard_address=None
    client_threaded_parallel =      Client(processes=False)
    print("                             DASK client_threaded          " , client_threaded_parallel)
    do_run(numIterations, dask_driver, "DASK thrd multi:", client=client_threaded_parallel)
    client_threaded_parallel.close()
    print("")

    client_threaded_single =        Client(processes=False, n_workers=1, threads_per_worker=1)
    print("                             DASK client_threaded_single   " , client_threaded_single)
    do_run(numIterations, dask_driver, "DASK thrd sing: ", client=client_threaded_single)
    client_threaded_single.close()
    print("")

    client_processes_parallel =     Client(processes=True)
    print("                             DASK client_processess        " , client_processes_parallel)
    do_run(numIterations, dask_driver, "DASK proc mult: ", client=client_processes_parallel)
    client_processes_parallel.close()
    print("")

    client_processes_single =       Client(processes=True, n_workers=1, threads_per_worker=1)
    print("                             DASK client_processes_single  " , client_processes_single)
    do_run(numIterations, dask_driver, "DASK proc sing: ", client=client_processes_single)
    client_processes_single.close()
    print("")

    # do small runs to warm up JIT
    do_run(WARM_ITER,     driver_cpu_jit_serial,   "CPU JIT warmup:   ")
    do_run(WARM_ITER,     driver_cpu_jit_parallel, "CPU JIT warmup:   ")
    do_run(numIterations, driver_cpu_jit_serial,   "CPU jit serial:   ")
    do_run(numIterations, driver_cpu_jit_parallel, "CPU jit parallel: ")

    do_run(numIterations, driver_cpu_nojit_serial, "CPU nojit serial: ")
