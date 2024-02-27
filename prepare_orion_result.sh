#!/bin/bash -e

AI_THRESHOLD=14.87
PYTHON=python3.8
NCU_DIR=/usr/local/NVIDIA-Nsight-Compute-2023.3
#NCU_DIR=/root/

source helper.sh

run_decorated_inference() {
    pre=$1
    post="> /dev/null 2>&1"
    if [[ $# -eq 2 ]]; then
        post=$2
    fi

    command="${pre} ${PYTHON} batched_inference.py --model ${model} --batch-size ${batch} --num-infer 1 --distribution_type closed --rps -1 ${post}"
    eval "$command" &
    ncu_pid=$!
    sleep 1

    readarray -t forked_pids < <(ps -eaf | grep batched_inference.py | grep -v "${NCU_DIR}" | grep -v "nsys" | grep -v grep | awk '{print $2}')
    if [[ ${#forked_pids[@]} > 1 ]]; then
        echo "Expected 1 batched_inference.py process! Seen != 1..."
        echo "Inspect using command: ' ps -eaf | grep batched_inference.py | grep -v "${NCU_DIR}" | grep -v "nsys" | grep -v grep'"
        exit 1
    fi

    # Wait for the model to load
    for pid in "${forked_pids[@]}"
    do
        while :
        do
            if [[ -f /tmp/${pid} ]]; then
                lt="$lt, $(cat /tmp/${pid})"
                rm -f /tmp/${pid}
                loaded_procs+=(${pid})
                break
            elif [[ -f /tmp/${pid}_oom ]]; then
                rm -f /tmp/${pid}_oom
                break
            fi

        done
    done

    # Start inference
    kill -SIGUSR1 ${forked_pids[@]}

    # Wait till the prefixed command completes
    while kill -0 ${ncu_pid} >/dev/null 2>&1; do sleep 1; done
}

profile_model() {
    device_type=$1
    model=$2
    batch=$3
    model_type=$4

    result_dir=$(pwd)/orion-results/${device_type}/${model}/batchsize-${batch}
    orion_dir=orion-fork
    kernel_file=${result_dir}/orion_input.csv
    config_file=${result_dir}/orion_config.json

    mkdir -p ${result_dir}
    rm -fr ${result_dir}/*

    echo "NCU Plain..."
    run_decorated_inference "${NCU_DIR}/ncu -f -o ${result_dir}/output_ncu --set detailed --nvtx --nvtx-include \"start/\""

    echo "NCU CSV..."
    run_decorated_inference "${NCU_DIR}/ncu -f --csv --set detailed --nvtx --nvtx-include \"start/\"" "> ${result_dir}/output_ncu.csv"

    echo "NSYS PROFILE..."
    run_decorated_inference "nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o ${result_dir}/output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi -f true -x true"

    echo "CONVERSION OF RESULT..."
    ${NCU_DIR}/ncu --csv --page raw -i ${result_dir}/output_ncu.ncu-rep > ${result_dir}/raw_ncu.csv
    sed -i '/^==PROF==/d' ${result_dir}/output_ncu.csv

    echo "CREATING ORION CONSUMABLE INPUT"
    ${PYTHON} ${orion_dir}/profiling/postprocessing/process_ncu.py --results_dir ${result_dir}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/get_num_blocks.py --results_dir ${result_dir} --device_type ${device_type}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/roofline_analysis.py --results_dir ${result_dir} --ai_threshold ${AI_THRESHOLD}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/generate_file.py \
        --input_file_name ${result_dir}/output_ncu_sms_roofline.csv \
        --output_file_name ${kernel_file} \
        --model_type ${model_type}


    num_kernels=$(($(wc -l < "${kernel_file}") - 1))

    sed -e "s|{{ model }}|${model}|" \
        -e "s|{{ kernel_file }}|${kernel_file}|" \
        -e "s|{{ num_kernels }}|${num_kernels}|" \
        -e "s|{{ batchsize }}|${batch}|" \
        orion_config.json.template > ${config_file}
}

if [[ $# -lt 4 ]]; then
    echo "Expected Syntax: '$0 <v100 | a100 | h100> model batchsize <vision | bert | transformer>"
    echo "Examples:"
    echo " $0 v100 vgg11 16 vision"
    exit 1
fi

profile_model $@

