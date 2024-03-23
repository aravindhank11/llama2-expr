c#!/bin/bash 

# Retrieve the starting line number from the command-line argument
start_line="$1"
device_id="$2"
size="$3"

# Check if both arguments are provided and valid
if [[ -z "$start_line" || ! "$start_line" =~ ^[0-9]+$ ]] || [[ ! "$device_id" =~ ^[0-3]$ ]] || [[ ! "$size" =~ ^[2-4]$ ]]; then
    echo "Usage: $0 <starting_line_number> <device_id (0-3)> <size (2-4)" >&2
    exit 1
fi

input_file="./test_combos_${size}.txt"

# Read lines starting from the specified line and ending 60 lines later
counter=0
while IFS= read -r line && [ $counter -lt 114 ]; do
    ((counter++))

    result_line=""

    read -ra parts <<< "$line"
    for part in "${parts[@]}"; do
        IFS='-' read -ra substrings <<< "$part"
        concatenated="${substrings[0]}-${substrings[1]}"
        result_line="$result_line vision-$concatenated"
    done
    model_mixes+=("$result_line")
done < <(sed -n "${start_line},+114p" "$input_file")

for model_mix in "${model_mixes[@]}"
do
    # ./run.sh --device-type a100 --device-id 0 --modes mps-uncap --distribution closed ${model_mix}
    cmd="./run.sh --device-type a100 --device-id ${device_id} --modes mps-uncap --duration 60 --distribution poisson --load-start 0.8 --load-end 1.0 ${model_mix}"
    echo "${cmd}"
    eval $cmd
done
