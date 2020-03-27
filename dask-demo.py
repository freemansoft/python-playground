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
import CalcCore

numIterations = 10000000
# warm up the jit
WARM_ITER = 5

# -------------------------------------------------------------------------------
#                     DASK cluster / worker calculations
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
#                     DASK
# -------------------------------------------------------------------------------

# DASK driver for each node - one in each task
# using parallel might give you 50% on threaded
def driver_dask_node(all_params):
    iteration_count = all_params.shape[0]
    result = np.zeros(iteration_count, dtype=CalcCore.atype)
    CalcCore.calc_cpu_jit_serial(result,iteration_count,all_params)
    return result

# -------------------------------------------------------------------------------
# Dask approach is different than GPU. Don't map out 10,000 function calls for DASK
# -------------------------------------------------------------------------------

# master dask adapter 
# a is an array of 1x3 arrays
def driver_dask(client, iteration_count, all_inputs):
    # have the worker nodes create the results array
    #result = np.zeros(iteration_count, dtype=atype)
    numDaskTasks = calc_dask_worker_threads(client)
    # split the list into segments one for each dask task - length may vary
    a_list = np.array_split(all_inputs,numDaskTasks)
    print("                             numDaskTask:", numDaskTasks, " each around:", a_list[0].shape[0])

    # scatter the lists (sequences?) of 1x3 arrays to the task nodes returns list of futures
    # returns list of futures
    scattered_future = client.scatter(a_list)
    # execute this function across the cluster against the scattered data
    calculated_future = client.map(driver_dask_node, scattered_future)
    # returns array of result arrays
    results_gathered = client.gather(calculated_future)
    # concatenate the arrays of result arrays into a single array
    results_unified = np.concatenate(results_gathered)
    return results_unified


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

if __name__=="__main__": 
    print("")
    print("")

    # open all of these with dashboards. Open and close in sequence so no conflicts
    # Can open all at once if disable dashboard:  dashboard_address=None
    client_threaded_parallel =      Client(processes=False)
    print("                             DASK client_threaded          " , client_threaded_parallel)
    CalcCore.do_run(numIterations, driver_dask, "DASK thrd multi:", client=client_threaded_parallel)
    client_threaded_parallel.close()
    print("")

    client_threaded_single =        Client(processes=False, n_workers=1, threads_per_worker=1)
    print("                             DASK client_threaded_single   " , client_threaded_single)
    CalcCore.do_run(numIterations, driver_dask, "DASK thrd sing: ", client=client_threaded_single)
    client_threaded_single.close()
    print("")

    client_processes_parallel =     Client(processes=True)
    print("                             DASK client_processess        " , client_processes_parallel)
    CalcCore.do_run(numIterations, driver_dask, "DASK proc mult: ", client=client_processes_parallel)
    client_processes_parallel.close()
    print("")

    client_processes_single =       Client(processes=True, n_workers=1, threads_per_worker=1)
    print("                             DASK client_processes_single  " , client_processes_single)
    CalcCore.do_run(numIterations, driver_dask, "DASK proc sing: ", client=client_processes_single)
    client_processes_single.close()
    print("")

    # run cpu twice - jit overhead on first run
    CalcCore.do_run(WARM_ITER,     CalcCore.driver_cpu_jit_parallel, "CPU: JIT warmup   ")
    CalcCore.do_run(WARM_ITER,     CalcCore.driver_cpu_jit_serial,   "CPU: JIT warmup   ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_jit_parallel, "CPU jit parallel: ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_jit_serial,   "CPU jit serial:   ")
    CalcCore.do_run(numIterations, CalcCore.driver_cpu_nojit_serial, "CPU serial nojit: ")
