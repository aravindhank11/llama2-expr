#!/bin/bash 
if [[ $# -lt 2 ]]; then
    echo "Expected Syntax: '$0 <ts|mps-uncap|mps-equi|mps-miglike|mig> num-procs"
    echo "Examples:"
    echo " $0 mig 3"
    exit 1
fi

mode=$1
num_procs=$2

source tie-breaker-venv/bin/activate
source helper.sh

assert_mig_status ${mode}
enable_mps_if_needed ${mode}
setup_mig_if_needed ${mode} ${num_procs}

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f llama-h100-${num_procs}_${mode}.csv &
stats_pid=$!

if [[ ${mode} == "mps-miglike" ]]; then
	    percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode} == "mps-equi" ]]; then
	        percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
fi

echo -e "\nRun: num_procs=${num_procs} batch_sizes=${batch_sizes[@]}\n"

cci_uuid=($(nvidia-smi -L | grep MIG- | awk '{print $6}' | sed 's/)//g'))
chunk_id=($(nvidia-smi -L | grep MIG- | awk '{print $2}' | sed 's/)//g'))
last=$((num_procs-1))

for (( c=0; c<${num_procs}; c++ ))
do
    if [[ ${mode} == "mig" ]]; then
        echo "Setting ${chunk_id[$c]}"
        export CUDA_VISIBLE_DEVICES=${cci_uuid[$c]}
    elif [[ ${mode} == "mps-miglike" || ${mode} == "mps-equi" ]]; then
        echo "Setting ${percent[$c]}%"
        export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${percent[$c]}
    else
        export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0
    fi

    if [[ $c -ne $last ]]; then
        python3 llama2.py &
    else
        python3 llama2.py
    fi
done

kill -INT ${stats_pid}

cleanup ${mode}
