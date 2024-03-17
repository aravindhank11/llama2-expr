#!/bin/bash -e

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

    command="${pre} \
        ${PYTHON} src/batched_inference_executor.py \
        --device-id ${device_id} \
        --model-type ${model_type} \
        --model ${model} \
        --batch-size ${batch} \
        --num-infer 1 \
        --distribution-type closed \
        --rps 0 \
        --tid 0 \
        ${post} &"
    eval "$command"
    ncu_pid=$!
    sleep 1

    readarray -t forked_pids < <(ps -eaf | grep batched_inference_executor.py | grep -v "${NCU_DIR}" | grep -v "nsys" | grep -v grep | awk '{print $2}')
    if [[ ${#forked_pids[@]} != 1 ]]; then
        echo "Expected 1 batched_inference_executor.py process! Seen != 1..."
        echo "Inspect using command: ' ps -eaf | grep batched_inference_executor.py | grep -v "${NCU_DIR}" | grep -v "nsys" | grep -v grep'"
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
    device_id=$2
    model=$3
    batch=$4
    model_type=$5

    if [[ (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
        echo "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        echo "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    if [[ ${model_type} != "llama" && ${model_type} != "vision" ]]; then
        echo "Allowed model types: llama and vision. Got: ${model_type}"
        exit 1
    fi

    result_dir=$(pwd)/orion-fork/results/${device_type}/${model}/batchsize-${batch}
    orion_dir=orion-fork
    kernel_file=${result_dir}/orion_input.csv

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
    sed -i '/^"ID","Process ID"/,$!d' ${result_dir}/output_ncu.csv

    echo "CREATING ORION CONSUMABLE INPUT"
    read -p "Enter AI_THRESHOLD by looking at ${result_dir}/output_ncu.ncu-rep: " ai_threshold
    ${PYTHON} ${orion_dir}/profiling/postprocessing/process_ncu.py --results_dir ${result_dir}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/get_num_blocks.py --results_dir ${result_dir} --device_type ${device_type}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/roofline_analysis.py --results_dir ${result_dir} --ai_threshold ${ai_threshold}
    ${PYTHON} ${orion_dir}/profiling/postprocessing/generate_file.py \
        --input_file_name ${result_dir}/output_ncu_sms_roofline.csv \
        --output_file_name ${kernel_file} \
        --model_type ${model_type}
}

if [[ $# -lt 5 ]]; then
    echo "Expected Syntax: '$0 <v100 | a100 | h100> <device_id> model batchsize <vision | bert | transformer>"
    echo "Examples:"
    echo " $0 v100 0 vgg11 16 vision"
    exit 1
fi

mv -v /root/orion /root/orion-bkp > /dev/null 2>&1 || :
profile_model $@
