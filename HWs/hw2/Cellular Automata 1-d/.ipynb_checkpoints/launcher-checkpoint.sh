#!/bin/sh
mpicc speedup.c -o tmp.out
N=10
m=8
n=8
res="res-"

for ((i=1; i<m; i++))
do
    for ((j=1; j<n; j++))
    do
        mpirun --oversubscribe -np $j ./tmp.out 100 $N >> $res$i
    done
    N=$N"0"
done
rm tmp.out
