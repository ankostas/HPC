#!/bin/sh
rm out-2.out
rm res-2.txt
mpicc send_names_2.c -o out-2.out
word="name"
for ((i=0; i <= 15; i++))
do
    echo $i
    mpirun --oversubscribe -np 4 out-2.out $word >> res-2.txt
    word=$word$word

done
