# -------------------------------------------------------------------------------
#                 THIS FILE IS NOT USEFUL AT THIS TIME
# -------------------------------------------------------------------------------


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

numIterations = 10000000
atype = np.float64
DASK_enabled = False

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

# function for dask 
# a is an array of 1x3 arrays
def invoke_dask(client, all_inputs):
    numDaskTasks = calc_dask_worker_threads(client)
    print("numDaskTask:", numDaskTasks)
    # split the list into segments one for each dask task
    a_list = np.array_split(all_inputs,numDaskTasks)
    # scatter the lists (sequences?) of 1x3 arrays to the task nodes returns list of futures
    # returns list of futures
    scattered_future = client.scatter(a_list)
    # execute this function across the cluster against the scattered data
    calculated_future = client.map(cpu_driver, scattered_future)
    # returns array of result arrays
    results_gathered = client.gather(calculated_future)
    # concatenate the arrays of result arrays into a single array
    results_unified = np.concatenate(results_gathered)
    # free cluster memory
    # return the single array results should be the same length as length f all_inputs
    return results_unified

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

# jit gives 20x performance
@jit
def math_on_cpu(param_triple):
   return param_triple[0]+param_triple[1]+param_triple[2]

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

# normal function to run on cpu
# also serialized to dask workers
@jit(parallel=True)
def calc_on_cpu(result, iteration_count, all_params):
    # prange tells jit this can be executed in parallel
    for i in prange(iteration_count):
        param_array = all_params[i]
        result[i] = math_on_cpu(param_array)


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

def cpu_driver(result, iteration_count, all_params):
    calc_on_cpu(result,iteration_count,all_params)


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Invoke dask with different clients if client supplied
# Invoke cpu and gpu if client not provided
def do_run(iterations, the_func, label, client=None):
    # create a one long 1D array for result
    result = np.zeros(iterations, dtype=atype)
    # create a one long 1D array with synthetic test data
    one_dimension = np.arange(iterations, dtype=atype)
    # create a 2D numIterations long array of 1x3 arrays so [numIterations[1,3]]
    triplets = np.vstack((one_dimension,one_dimension,one_dimension)).T
    start = timer()
    if client is None:
        the_func(result, iterations, triplets)
        #uncomment this line to see parallel execution optimization
        #the_func.parallel_diagnostics(level=2)
    else : 
        print("                      ", client, end=" ")
        result = the_func(client, iterations, triplets)
    result_len = len(result)
    span = timer() - start
    print(label, f"{span:,.4f}"," Iterations:",f"{iterations:,}", " dtype=", atype.__name__, "numcalc:", f"{result_len:,}", "sum:", np.sum(result,dtype=atype))


if __name__=="__main__": 
    client_threaded =         Client(processes=False, dashboard_address=None)
    print("DASK client_threaded          " , client_threaded)
    client_threaded_single =  Client(processes=False, dashboard_address=None, n_workers=1, threads_per_worker=1)
    print("DASK client_threaded_single   " , client_threaded_single)

    client_processes =        Client(processes=True,  dashboard_address=8787)
    print("DASK client_processess        " , client_processes)
    client_processes_single = Client(processes=True,  dashboard_address=None, n_workers=1, threads_per_worker=1)
    print("DASK client_processes_single  " , client_processes_single)
    
    print("")
    # run cpu twice - jit overhead on first run
    do_run(numIterations, cpu_driver,        "CPU:            ")
    do_run(numIterations, cpu_driver,        "CPU:            ")
    do_run(numIterations, invoke_dask, "DASK proc mult: ", client=client_processes)
    do_run(numIterations, invoke_dask, "DASK proc sing: ", client=client_processes_single)
    do_run(numIterations, invoke_dask, "DASK thrd multi:", client=client_threaded)
    do_run(numIterations, invoke_dask, "DASK thrd sing: ", client=client_threaded_single)

    print("")
    print("")
    print("")

    # import time
    # time.sleep(10.0)
    # do_run(numIterations, invoke_dask,"DASK proc mult: ", client=client_processes)
    # time.sleep(1.0)
    # do_run(numIterations, invoke_dask,"DASK proc mult: ", client=client_processes)
    # while True:
    #     print("Hanging around so you can hit the dashboard http://localhost:8787")
    #     time.sleep(4.0)

