#!/bin/bash

# Retrieve the starting line number from the command-line argument
start_line="$1"
device_id="$2"

# Check if both arguments are provided and valid
if [[ -z "$start_line" || ! "$start_line" =~ ^[0-9]+$ ]] || [[ ! "$device_id" =~ ^[0-3]$ ]]; then
    echo "Usage: $0 <starting_line_number> <device_id (0-3)>" >&2
    exit 1
fi

input_file="./model-formulation/job_mixes/combos_2_nlp.txt"

# Read lines starting from the specified line and ending 50 lines later
counter=0
while IFS= read -r line && [ $counter -lt 10 ]; do
    model_mixes+=("$line")
    ((counter++))
done < <(sed -n "${start_line},+10p" "$input_file")

for model_mix in "${model_mixes[@]}"
do
    # ./run.sh --device-type a100 --device-id 0 --modes mps-uncap --distribution closed ${model_mix}
    cmd="./run.sh --device-type a100 --device-id ${device_id} --modes mps-uncap --duration 300 --distribution closed ${model_mix}"
    echo "${model_mix}"
    # eval $cmd
done
