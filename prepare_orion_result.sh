#!/bin/bash -xe
if [[ $# -lt 4 ]]; then
    echo "Expected Syntax: '$0 <v100 | a100 | h100> model batchsize <vision | bert | transformer>"
    echo "Examples:"
    echo " $0 v100 vgg11 16 vision"
    exit 1
fi

device_type=$1
model=$2
batch=$3
model_type=$4

AI_THRESHOLD=14.87
DEVICE_NAME="cuda:0"
NCU_DIR=/opt/nvidia/nsight-compute/2023.1.1
RESULT_DIR=orion-results/${device_type}/${model}/batchsize-${batch}
ORION_DIR=orion-fork

mkdir -p ${RESULT_DIR}
rm -fr ${RESULT_DIR}/*
source helper.sh
source ${VENV}/bin/activate

run_decorated_inference() {
    pre=$1
    post="> /dev/null 2>&1"
    if [[ $# -eq 2 ]]; then
        post=$2
    fi

    command="${pre} python3 batched_inference.py --model ${model} --batch-size ${batch} --num-infer 1 ${post}"
    eval "$command" &
    ncu_pid=$!
    sleep 1

    readarray -t forked_pids < <(ps -eaf | grep batched_inference.py | grep -v "nsys profile" | grep -v "nsight-compute" | grep -v grep | awk '{print $2}')
    if [[ ${#forked_pids[@]} > 1 ]]; then
        echo "Expected 1 batched_inference.py process! Seen != 1..."
        echo "Inspect using command: ' ps -eaf | grep batched_inference.py | grep -v "nsight-compute" | grep -v grep'"
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

echo "NCU Plain..."
run_decorated_inference "${NCU_DIR}/ncu -f -o ${RESULT_DIR}/output_ncu --set detailed --nvtx --nvtx-include \"start/\""

echo "NCU CSV..."
run_decorated_inference "${NCU_DIR}/ncu -f --csv --set detailed --nvtx --nvtx-include \"start/\"" "> ${RESULT_DIR}/output_ncu.csv"

echo "NSYS PROFILE..."
run_decorated_inference "nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o ${RESULT_DIR}/output_nsys --cudabacktrace=true --capture-range=cudaProfilerApi -f true -x true"

echo "CONVERSION OF RESULT..."
${NCU_DIR}/ncu --csv --page raw -i ${RESULT_DIR}/output_ncu.ncu-rep > ${RESULT_DIR}/raw_ncu.csv
sed -i '/^==PROF==/d' ${RESULT_DIR}/output_ncu.csv

echo "CREATING ORION CONSUMABLE INPUT"
python ${ORION_DIR}/profiling/postprocessing/process_ncu.py --results_dir ${RESULT_DIR}
python ${ORION_DIR}/profiling/postprocessing/get_num_blocks.py --results_dir ${RESULT_DIR} --device_type ${device_type}
python ${ORION_DIR}/profiling/postprocessing/roofline_analysis.py --results_dir ${RESULT_DIR} --ai_threshold ${AI_THRESHOLD}
python ${ORION_DIR}/profiling/postprocessing/generate_file.py \
    --input_file_name ${RESULT_DIR}/output_ncu_sms_roofline.csv \
    --output_file_name ${RESULT_DIR}/orion_input.csv \
    --model_type vision
