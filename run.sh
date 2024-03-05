#!/bin/bash

DURATION=180
DISTRIBUTION=poisson
MODES=("mps-uncap" "orion" "ts")
start=0.1
end=1.5
step=0.1

source helper.sh
WS=$(git rev-parse --show-toplevel)
DOCKER_WS=/root/$(basename ${WS})

multiply_and_round() {
    local result=$(echo "$1 * $2" | bc)
    if [[ $# -eq 3 ]]; then
        printf "%.$3f\n" "$result"
    else
        printf "%.2f\n" "$result"
    fi
}

divide_and_round() {
    local result=$(echo "scale=2; $1 / $2" | bc)
    printf "%.2f\n" "$result"
}

generate_closed_loop_load() {
    params=""
    for (( i=0; i<${num_procs}; i++ ))
    do
        params="$params ${model_types[$i]}-${models[$i]}-${batch_sizes[$i]}-closed-0"
    done
    cmd=".${DOCKER} exec -it ${TIE_BREAKER_CTR} bash -c cd ${DOCKER_WS} && ./run_expr.sh ${DURATION} ${device_type} mps-uncap 1 ${params}"
    echo "Running closed loop experiment:"
    echo "Command used: ${cmd}"
    eval ${cmd}
}

generate_distribution_load() {
    for mode in ${MODES[@]}
    do
        if [[ ${RUNNING_IN_DOCKER} -ne 1 ]]; then
            assert_mig_status ${mode} ${device_id}
            enable_mps_if_needed ${mode} ${device_id}
            setup_mig_if_needed ${mode} ${device_id} ${num_procs}
        fi

        if [[ ${mode} != "orion" ]]; then
            docker_prefix="${DOCKER} exec -it ${TIE_BREAKER_CTR} bash -c cd ${DOCKER_WS} && "
        else
            docker_prefix=""
        fi

        for ((i = $(multiply_and_round ${start} 10 0); i <= $(multiply_and_round ${end} 10 0); i++)); do
            ratio=$(echo "$start + ($i - 1) * $step" | bc)
            params=""
            for ((j=0; j<${num_procs}; j++))
            do
                mul=$(multiply_and_round ${ratio} ${rps[$j]})
                params="$params ${model_types[$j]}-${models[$j]}-${batch_sizes[$j]}-${DISTRIBUTION}-${mul}"
            done
            cmd="${docker_predix} ./run_expr.sh ${duration} ${device_type} ${mode} ${ratio} ${params}"
            echo "${mode} ${ratio} ${cmd}"
            #eval ${cmd} >> print_outs.txt 2>&1
        done

        if [[ ${RUNNING_IN_DOCKER} -ne 1 ]]; then
            cleanup ${mode} ${device_id}
        fi
    done
}

print_help() {
    echo "Expected Syntax:"
    echo "  $0 <v100|h100|a100> <model_type1-model1-batch_size1> <model_type2-model2-batch_size2> ... "
    echo -e "\n"

    echo "NOTE:"
    echo "  Supported model_types are: 'vision', 'llama'"

    echo "Examples:"
    echo " $0 v100 vision-vgg19-32"
    echo " $0 a100 vision-vgg19-32 vision-mobilenet_v2-1"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
    exit 1
}

if [[ $# -lt 2 ]]; then
    print_help
fi

device_type=$1
shift
model_run_params=( "$@" )
num_procs=${#model_run_params[@]}

if [[ ${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100" ]]; then
    echo "Invalid device_type: ${device_type}"
    exit 1
fi

pattern="^[^-]+-[^-]+-[^-]+$"
model_types=()
models=()
batch_sizes=()
for (( i=0; i<${num_procs}; i++ ))
do
    element=${model_run_params[$i]}
    if ! [[ "${element}" =~ $pattern ]]; then
        echo "Improper input: Unable to get batch size for ${element}"
        echo "Expected: ModelType-Model-BatchSize"
        echo "Example: vision-densenet121-32"
        exit 1
    fi
    model_types[$i]=$(echo $element | cut -d'-' -f1)
    models[$i]=$(echo $element | cut -d'-' -f2)
    batch_sizes[$i]=$(echo $element | cut -d'-' -f3)
done


# Generate closed loop load
#generate_closed_loop_load

# Gather metrics and generate RPS values
result_base=$(IFS=- ; echo "${models[*]}")
for i in "${!models[@]}"; do
    concatenated_string="${concatenated_string}${models[$i]}-${batch_sizes[$i]}-closed_"
done
result_id=${concatenated_string%_}
tput_csv=results/hetro/${result_base}/${result_id}/tput.csv
read -r -a tput_csv_string <<< "$(tail -n 1 $tput_csv | awk -v n=${num_procs} -F ',' '{split($0, a); for(i=NF-n+1; i<=NF; i++) printf "%s%s", a[i], (i==NF) ? "" : FS}')"
IFS=',' read -r -a tput_metrics <<< "$tput_csv_string"
echo "TPUT: ${tput_metrics[@]}"
rps=()
for ((i=0; i<${num_procs}; i++))
do
    met=$(divide_and_round ${tput_metrics[$i]} ${batch_sizes[$i]})
    rps+=(${met})
done
echo "RPS: ${rps[@]}"

# Generate distribution load
generate_distribution_load
