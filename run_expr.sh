if [[ $# -lt 2 ]]; then
    echo "Expected Syntax: '$0 <ts|mps-uncap|mps-equi|mps-miglike|mig> <model1-batch1> <model2-batch2> ... <modeln-batchn>'"
    echo "Examples:"
    echo " $0 ts vgg19-32"
    echo " $0 ts vgg19-32 mobilenet_v2-1"
    echo " $0 mps-equi vgg19-32 mobilenet_v2-1"
    echo " $0 mps-uncap densenet121-32 inception_v3-32"
    echo " $0 mig densenet121-32 inception_v3-32"
    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
    exit 1
fi

source helper.sh
if [[ ! -d ${VENV} ]]; then
    echo "Look at README.md and install python3 virtual environment"
    exit 1
fi
source ${VENV}/bin/activate

mode=$1
shift
models_and_batch_sizes=( "$@" )
num_procs=${#models_and_batch_sizes[@]}

batch_sizes=()
models=()
for (( i=0; i<${num_procs}; i++ ))
do
    element=${models_and_batch_sizes[$i]}
    if [[ ${element} != *"-"* ]]; then
        echo "Improper input: Unable to get batch size"
        echo "Expected: Model-BatchSize"
        echo "Example: densenet121-32"
        exit 1
    fi
    models[$i]=$(echo $element | cut -d'-' -f1)
    batch_sizes[$i]=$(echo $element | cut -d'-' -f2)
done


echo "mode=$mode batch_sizes=${batch_sizes[@]} models=${models[@]} num_procs=$num_procs"
echo "models=${models[@]}"

############################################################ HUGE ASSUMPTION
DEVICE_NAME=v100
RESULT_FILE=${models[0]}-${DEVICE_NAME}-${mode}-${num_procs}.csv

assert_mig_status ${mode}
enable_mps_if_needed ${mode}
setup_mig_if_needed ${mode} ${num_procs}


if [[ ${mode} == "mps-miglike" ]]; then
    percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
elif [[ ${mode} == "mps-equi" ]]; then
    percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
fi

echo -e "\nRun: num_procs=${num_procs} batch_sizes=${batch_sizes[@]}\n"

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

    python3 batched_inference.py --model ${models[$c]} --batch-size ${batch_sizes[$c]} > /dev/null 2>&1 &
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
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f ${RESULT_FILE} &
stats_pid=$!

# Start inference
echo "Starting inference on ${loaded_procs[@]}"
kill -SIGUSR1 ${loaded_procs[@]}

# Run experiment for 10s
sleep 10

# Stop inference
kill -SIGUSR2 ${loaded_procs[@]}

# Stop collecting state
kill -SIGINT ${stats_pid}

# Start recording now
tput=""
pkl_files=()
for pid in "${loaded_procs[@]}"
do
    while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    if [[ -f /tmp/${pid} ]]; then
        tput="$tput, $(cat /tmp/${pid})"
        pkl_files+=("/tmp/${pid}.pkl")
        rm -f /tmp/${pid}
    elif [[ -f /tmp/${pid}_oom ]]; then
        rm -f /tmp/${pid}_oom
    fi
done


cleanup ${mode}

while taskset -c 0 kill -0 ${stats_pid} >/dev/null 2>&1; do sleep 1; done

tput="${tput/, /}"
IFS=", " read -ra numbers <<< "$tput"
tput_sum=0
for number in "${numbers[@]}"; do
    # Add each number to the sum using bc for floating-point arithmetic
    tput_sum=$(echo "$tput_sum + $number" | bc)
done
latency=$(python3 latency_stats.py ${pkl_files[@]})
cpu_mem=$(./stats.sh ${RESULT_FILE})
echo ""

echo "Throughputs: ${tput}"
echo "Total Throughput: ${tput_sum}"
echo ""

echo "min_lat, 50th_lat, 90th_lat, 99_lat, max_lat"
echo "${latency}"
echo ""

echo "min_compute, avg_compute, max_compute, min_mem, avg_mem, max_mem"
echo "${cpu_mem}"
