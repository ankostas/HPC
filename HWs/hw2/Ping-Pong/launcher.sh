#!/bin/sh
rm out.out
rm res.txt
mpicc send_names.c -o out.out
name="name"
mpirun --oversubscribe -np 4 out.out $name >> res.txt


