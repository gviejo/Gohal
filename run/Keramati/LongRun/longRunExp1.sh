#!/bin/bash

niterations=10

for i in $(seq $niterations)
do
    echo 'Bash iterations' $i
    python exp1.py -o LongRun/exp1run$i.pickle
done

