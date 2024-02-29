#!/bin/bash

EXPERIMENT_DURATION=10

if [[ $# -lt 3 ]]; then
    echo "Expected Syntax: '$0 <v100|h100|a100> <orion|ts|mps-uncap|mps-equi|mps-miglike|mig> <model_type1-model1-batch1-distribution_type1-rps1> <model_type2-model2-batch2-distribution_type2-rps2> ... '"
    echo "Examples:"
    echo " $0 v100 ts vision-vgg19-32-point-10"
    echo " $0 v100 ts vision-vgg19-32-poisson-10 vision-mobilenet_v2-1-closed-100"
    echo " $0 h100 mps-equi vision-vgg19-32-point-10 vision-mobilenet_v2-1-point-10"
    echo " $0 a100 mps-uncap llama-llama-32-poisson-100 vision-inception_v3-32-closed-23"
    echo " $0 a100 mig vision-densenet121-32-closed-10 vision-inception_v3-32-poisson-100"
    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
    exit 1
fi

export USE_SUDO=1
source helper.sh
if [[ ! -d ${VENV} ]]; then
    echo "Look at README.md and install python3 virtual environment"
    exit 1
fi
source ${VENV}/bin/activate

device_name=$1
shift
mode=$1
shift
model_run_params=( "$@" )
num_procs=${#model_run_params[@]}

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
        echo "Improper input: Unable to get batch size for ${element}"
        echo "Expected: ModelType-Model-BatchSize-DistributionType-RPS"
        echo "Example: vision-densenet121-32-point-30"
        exit 1
    fi
    model_types[$i]=$(echo $element | cut -d'-' -f1)
    models[$i]=$(echo $element | cut -d'-' -f2)
    batch_sizes[$i]=$(echo $element | cut -d'-' -f3)
    distribution_types[$i]=$(echo $element | cut -d'-' -f4)
    rps[$i]=$(echo $element | cut -d'-' -f5)
done


echo "mode=${mode} num_procs=${num_procs}"
echo "model_types=${model_types[@]}"
echo "models=${models[@]}"
echo "batch_sizes=${batch_sizes[@]}"
echo "distribution_types=${distribution_types[@]}"
echo "rps=${rps[@]}"

cpu_mem_metrics_file=/tmp/${models[0]}-${device_name}-${mode}-${num_procs}.csv
result_id=$(IFS=- ; echo "${model_run_params[*]}")
result_base=$(IFS=- ; echo "${models[*]}")
result_dir=results/hetro/${result_base}/${result_id}
mkdir -p ${result_dir}
per_model_stats_file=${result_dir}/per_model_stats
cpu_mem_stats_file=${result_dir}/cpu_mem_stats_file

run_orion_expr() {
    # Setup orion container
    ./setup_orion_container.sh

    # Start recording metrics
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f ${cpu_mem_metrics_file} &
    stats_pid=$!

    set -x

    # Run the experiment
    ws=$(git rev-parse --show-toplevel)
    docker_ws=$(basename ${ws})
    sudo docker exec -it orion bash -c \
        "cd /root/${docker_ws} && LD_PRELOAD='/root/orion/src/cuda_capture/libinttemp.so' python3.8 orion_scheduler.py --device-type ${device_name} --duration ${EXPERIMENT_DURATION} ${model_run_params[$@]}"
    sudo docker cp orion:/tmp/*.pkl ${tmpdir}

    # Stop collecting state
    kill -SIGINT ${stats_pid}

    # Stop cpu mem stats process and collect cpu mem stats
    while taskset -c 0 kill -0 ${stats_pid} >/dev/null 2>&1; do sleep 1; done

    # Collect the pickle files
    tmpdir=$(mktemp -d)
    pkl_files=()
    for f in $(ls ${tempdir})
    do
        pkl_files+=($f)
    done
}

run_other_expr() {
    assert_mig_status ${mode}
    enable_mps_if_needed ${mode}
    setup_mig_if_needed ${mode} ${num_procs}


    if [[ ${mode} == "mps-miglike" ]]; then
        percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode} == "mps-equi" ]]; then
        percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
    fi

    cci_uuid=($(nvidia-smi -L | grep MIG- | awk '{print $6}' | sed 's/)//g'))
    chunk_id=($(nvidia-smi -L | grep MIG- | awk '{print $2}' | sed 's/)//g'))

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

        python3 batched_inference.py \
            --model-type ${model_types[$c]} \
            --model ${models[$c]} \
            --batch-size ${batch_sizes[$c]} \
            --distribution_type ${distribution_types[$c]} \
            --rps ${rps[$c]} > /dev/null 2>&1 &
    done

    # Wait till all processes are up
    while :
    do
        readarray -t forked_pids < <(ps -eaf | grep batched_inference.py | grep -v grep | awk '{print $2}' )
        if [[ ${#forked_pids[@]} -eq ${num_procs} ]]; then
            break
        fi
        echo "Waiting for ${num_procs} processes to start: ${forked_pids[@]}"
        echo "${forked_pids[@]}"
        sleep 1
    done

    # Wait till all pids have loaded their models
    lt=""
    loaded_procs=()
    echo ${forked_pids[@]}
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

    # Start recording metrics
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f ${cpu_mem_metrics_file} &
    stats_pid=$!

    # Start inference
    echo "Starting inference on ${loaded_procs[@]}"
    kill -SIGUSR1 ${loaded_procs[@]}

    # Run experiment for 10s
    sleep ${EXPERIMENT_DURATION}

    # Stop inference
    kill -SIGUSR2 ${loaded_procs[@]}

    # Stop collecting state
    kill -SIGINT ${stats_pid}

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


    cleanup ${mode}

    # Stop cpu mem stats process and collect cpu mem stats
    while taskset -c 0 kill -0 ${stats_pid} >/dev/null 2>&1; do sleep 1; done
}

compute_stats()
{
    for pkl_file in ${pkl_files[@]}
    do
        output=$(python3 stats.py ${mode} ${pkl_file})
        header=$(echo "$output" | head -n 1)
        stats=$(echo "$output" | tail -n 1)
        if [[ ! -f ${per_model_stats_file} ]]; then
            echo -e "$header" > ${per_model_stats_file}
        fi
        echo -e "$stats" >> ${per_model_stats_file}
    done


    output=$(./stats.sh ${mode} ${cpu_mem_metrics_file})
    header=$(echo "$output" | head -n 1)
    stats=$(echo "$output" | tail -n 1)
    if [[ ! -f ${cpu_mem_stats_file} ]]; then
        echo "$header" > ${cpu_mem_stats_file}
    fi
    echo "$stats" >> ${cpu_mem_stats_file}
}

if [[ ${mode} == "orion" ]]; then
    run_orion_expr
else
    run_other_expr
fi

compute_stats

