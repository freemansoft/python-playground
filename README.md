



# cuda-demo numbers

## Algorithm
math.log(param_triple[0])*math.log(param_triple[1])*math.log(param_triple[2])

## Z820 2 E5-26
```
CPU serial:     1.1330  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
CPU parallel:   0.0809  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
GPU grid:       0.1579  Iterations: 10,000,000  dtype= float64 numcalc: sum( 10,000,000 ): 34986983045.74297
```