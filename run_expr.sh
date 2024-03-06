#!/bin/bash

print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-paramete ..."
    echo "Options:"
    echo "  --device-type DEVICE_TYPE                   v100, a100, h100                                   (required)"
    echo "  --device-id   DEVICE_ID                     0, 1, 2, ..                                        (required)"
    echo "  --mode        MODE_OF_EXPR                  orion, ts, mps-uncap, mps-equi, mps-miglike, mig   (required)"
    echo "  --duration    DURATION_OF_EXPR_IN_SECONDS   10                                                 (default 120)"
    echo "  --load        LOAD_INDICATOR                0.5, 0.2, 1                                        (default 1)"
    echo "  -h, --help                                  Show this help message"
    echo -e "\n"

    echo "NOTE:"
    echo "  Load generation is done via Distribution Type and RPS"
    echo -e "\n"

    echo "Examples:"
    echo " $0 --device-type v100 --device-id 0 --duration 10 --mode ts --load 1 vision-vgg19-32-point-10"
    echo " $0 --device-type v100 --device-id 1 --duration 20 --mode ts --load 0.8 vision-vgg19-32-poisson-10 vision-mobilenet_v2-1-closed-100"
    echo " $0 --device-type h100 --device-id 0 --duration 100 --mode mps-equi --load 0.4 vision-vgg19-32-point-10 vision-mobilenet_v2-1-point-10"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
}

# Parse arguments
model_run_params=()
load=1
duration=180
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-type)
            device_type="$2"
            shift 2
            ;;
        --device-id)
            device_id="$2"
            shift 2
            ;;
        --duration)
            duration="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --load)
            load="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            model_run_params+=("$1")
            shift
            ;;
    esac
done

export USE_SUDO=1
num_procs=${#model_run_params[@]}
source helper.sh

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

if [[ -z ${duration} || ! ${duration} =~ ^[+-]?[0-9]*\.?[0-9]+$ ]]; then
    echo "duration_in_seconds must be a float. Got ${duration}"
    print_help
    exit 1
fi

if [[ ${mode} != "mps-uncap" && ${mode} != "mps-equi" && ${mode} != "mps-miglike" && ${mode} != "mig" && ${mode} != "ts" && ${mode} != "orion" ]]; then
    echo "Invalid mode: ${mode}"
    print_help
    exit 1
fi

if [[ -z ${load} || ! ${load} =~ ^[+-]?[0-9]*\.?[0-9]+$ ]]; then
    echo "load must be a float. Got ${load}"
    print_help
    exit 1
fi

if [[ ${num_procs} -eq 0 ]]; then
    echo "Need at least 1 model configuration to run"
    print_help
    exit 1
fi

model_types=()
models=()
batch_sizes=()
distribution_types=()
rps=()
pattern="^[^-]+-[^-]+-[^-]+-[^-]+-[^-]+$"
for (( i=0; i<${num_procs}; i++ ))
do
    element=${model_run_params[$i]}
    if ! [[ "${element}" =~ $pattern ]]; then
        echo "Expected: ModelType-Model-BatchSize-DistributionType-RPS"
        echo "Got: ${element}"
        echo "Example: vision-densenet121-32-point-30"
        exit 1
    fi
    model_types[$i]=$(echo $element | cut -d'-' -f1)
    models[$i]=$(echo $element | cut -d'-' -f2)
    batch_sizes[$i]=$(echo $element | cut -d'-' -f3)
    distribution_types[$i]=$(echo $element | cut -d'-' -f4)
    if [[ ${distribution_types[$i]} != "poisson" && ${distribution_types[$i]} != "point" && ${distribution_types[$i]} != "closed" ]]; then
        echo "Invalid distribution_type: ${distribution_types[$i]}"
        print_help
        exit 1
    fi

    if [[ ${model_types[$i]} != "llama" && ${model_types[$i]} != "vision" ]]; then
        echo "Allowed model types: llama and vision. Got: ${model_types[$i]} from ${element}"
        exit 1
    fi

    rps[$i]=$(echo $element | cut -d'-' -f5)
done


echo "mode=${mode} num_procs=${num_procs}"
echo "model_types=${model_types[@]}"
echo "models=${models[@]}"
echo "batch_sizes=${batch_sizes[@]}"
echo "distribution_types=${distribution_types[@]}"
echo "rps=${rps[@]}"

result_base=$(IFS=- ; echo "${models[*]}")
for i in "${!models[@]}"; do
    concatenated_string="${concatenated_string}${models[$i]}-${batch_sizes[$i]}-${distribution_types[$i]}_"
done
result_id=${concatenated_string%_}
result_dir=results/hetro/${result_base}/${result_id}
mkdir -p ${result_dir}
cpu_mem_stats_file=${result_dir}/cpu_mem_stats_file

run_orion_expr() {
    # Setup directory to collect stats
    tmpdir=$(mktemp -d)

    # Setup orion container
    setup_orion_container ${device_id}

    # Run the experiment
    model_defn_string="${model_run_params[*]}"
    cmd="${DOCKER} exec -it ${ORION_CTR} bash -c \
        \"LD_PRELOAD='/root/orion/src/cuda_capture/libinttemp.so' \
        python3.8 src/orion_scheduler.py \
        --device-type ${device_type} \
        --duration ${duration} \
        ${model_defn_string}\""
    eval ${cmd}

    # Copy the results
    rm -f ${tmpdir}/*
    ${DOCKER} exec ${ORION_CTR} sh -c "ls /tmp/*.pkl" | while read -r file; do
        ${DOCKER} cp orion:${file} ${tmpdir}
    done

    # Collect the pickle files
    pkl_files=()
    for f in $(ls ${tmpdir})
    do
        pkl_files+=("${tmpdir}/${f}")
    done
}

run_other_expr() {
    assert_mig_status ${mode} ${device_id}
    enable_mps_if_needed ${mode} ${device_id}
    setup_mig_if_needed ${mode} ${device_id} ${num_procs}

    if [[ ${mode} == "mps-miglike" ]]; then
        percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode} == "mps-equi" ]]; then
        percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
    fi

    cci_uuid=($(nvidia-smi -L | grep MIG- | awk '{print $6}' | sed 's/)//g'))
    chunk_id=($(nvidia-smi -L | grep MIG- | awk '{print $2}' | sed 's/)//g'))
    cmd_arr=()

    for (( c=0; c<${num_procs}; c++ ))
    do
        if [[ ${mode} == "mig" ]]; then
            echo "Setting ${chunk_id[$c]} for ${models[$c]}"
            export CUDA_VISIBLE_DEVICES=${cci_uuid[$c]}
        elif [[ ${mode} == "mps-miglike" || ${mode} == "mps-equi" ]]; then
            echo "Setting ${percent[$c]}% for ${models[$c]}"
            export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
            export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${percent[$c]}
        else
            export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0
        fi

        cmd="${DOCKER} exec -d -it ${TIE_BREAKER_CTR} \
            python3 src/batched_inference_executor.py  \
            --device-id ${device_id} \
            --model-type ${model_types[$c]} \
            --model ${models[$c]} \
            --batch-size ${batch_sizes[$c]} \
            --distribution_type ${distribution_types[$c]} \
            --rps ${rps[$c]} \
            --tid ${c}"
        eval $cmd
        cmd_arr+=("${cmd}")
    done
    sleep 1

    readarray -t forked_pids < <(ps -eaf | grep batched_inference_executor.py | grep -v grep | grep -v docker | awk '{print $2}')
    if [[ ${#forked_pids[@]} -ne ${num_procs} ]]; then
        echo "Expected ${num_procs} processes. But found ${#forked_pids[@]}}"
        echo "Examine commands: "
        for cmd in "${cmd_arr[@]}"
        do
            echo "  ${cmd}"
        done
        exit 1
    fi


    # Wait till all pids have loaded their models
    lt=""
    loaded_procs=()
    echo ${forked_pids[@]}
    i=0
    for pid in "${forked_pids[@]}"
    do
        load_ctr=100
        while [[ ${load_ctr} -gt 0 ]];
        do
            if [[ -f /tmp/${pid} ]]; then
                lt="$lt, $(cat /tmp/${pid})"
                rm -f /tmp/${pid}
                loaded_procs+=(${pid})
                break
            fi

            if ! kill -0 "${pid}" &> /dev/null; then
                echo "Process no longer alive"
                load_ctr=0
                break
            fi
            ((load_ctr--))
            sleep 1
        done

        if [[ ${load_ctr} -eq 0 ]]; then
            echo "Inspect execution of '${cmd_arr[$i]}'"
            exit 1
        fi
        ((i++))
    done

    # Start inference
    echo "Starting inference on ${loaded_procs[@]}"
    kill -SIGUSR1 ${loaded_procs[@]}

    # Run experiment for experiment duration
    sleep ${duration}

    # Stop inference
    kill -SIGUSR2 ${loaded_procs[@]}

    # Start recording now
    pkl_files=()
    for pid in "${loaded_procs[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
        pkl_file="/tmp/${pid}.pkl"
        if [[ -f ${pkl_file} ]]; then
            pkl_files+=(${pkl_file})
        elif [[ -f /tmp/${pid}_oom ]]; then
            rm -f /tmp/${pid}_oom
        fi
    done

    cleanup ${mode} ${device_id}
}

compute_stats()
{
    cmd="${DOCKER} exec -it ${TIE_BREAKER_CTR} \
        python3 src/stats.py \
        --mode ${mode} \
        --load ${load} \
        --result_dir ${result_dir} \
        ${pkl_files[@]}"
    eval ${cmd}
}

${SUDO} nvidia-smi -i ${device_id} -pm ENABLED
setup_tie_breaker_container ${device_id}
if [[ ${mode} == "orion" ]]; then
    run_orion_expr
else
    run_other_expr
fi

compute_stats
