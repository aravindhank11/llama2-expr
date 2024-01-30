#!/bin/bash
file=$1


if [[ -z $file ]]; then
    echo "Give a filename"
    exit 1
fi


mem=$(cat ${file} | awk '{print $((NF-1))}' | sort -n | tail -1)
compute=$(cat ${file} | awk '{print $((NF-9))}' | sort -n | tail -1)

echo "${file} ${compute} ${mem}"

