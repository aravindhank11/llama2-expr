#!/bin/bash
mode=$1
file=$2


if [[ -z $file ]]; then
    echo "Give a filename"
    exit 1
fi

mem=($(tail -n +2 "${file}" | awk '{print $(NF-1)}'))
compute=($(tail -n +2 "${file}" | awk '{print $(NF-9)}'))

avg_mem=0
min_mem=${mem[0]}
max_mem=${mem[0]}
for m in "${mem[@]}"
do
	avg_mem=$((avg_mem + m))
    if [[ $m -lt $min_mem ]]; then
        min_mem=$m
    fi
    if [[ $m -gt $max_mem ]]; then
        max_mem=$m
    fi
done
avg_mem=$(echo "scale=2; $avg_mem / ${#mem[@]}" | bc)


avg_compute=0
min_compute=${compute[0]}
max_compute=${compute[0]}
for c in "${compute[@]}"
do
	avg_compute=$((avg_compute + c))
    if [[ $c -lt $min_compute ]]; then
        min_compute=$c
    fi
    if [[ $c -gt $max_compute ]]; then
        max_compute=$c
    fi
done
avg_compute=$(echo "scale=2; $avg_compute / ${#compute[@]}" | bc)

echo "mode, compute_p0, compute_p50, compute_p100, mem_p0, mem_p50, mem_p100"
echo "${mode}, ${min_compute}, ${avg_compute}, ${max_compute}, ${min_mem}, ${avg_mem}, ${max_mem}"
