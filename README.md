



# cuda-demo numbers

## Algorithm
math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

## Z820 2X E5-2640 v2 @ 2.00GHz 2x8C
```
CPU serial:     1.1330  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
CPU parallel:   0.0809  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
GPU grid:       0.1579  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
```

```
DASK thrd multi: 2.3464  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
DASK thrd sing:  1.7812  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
DASK proc mult:  1.9312  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
DASK proc sing:  3.9343  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297

CPU serial:      1.3242  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
CPU parallel:    0.0716  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
```