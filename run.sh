#!/bin/bash -e

MODES=("mps-uncap" "orion" "ts")

source helper.sh
WS=$(git rev-parse --show-toplevel)
DOCKER_WS=~/$(basename ${WS})

generate_closed_loop_load() {
    get_closed_loop_tput_csv

    if [[ -f ${tput_csv_string} ]]; then
        return
    fi

    params=""
    for (( i=0; i<${num_procs}; i++ ))
    do
        params="$params ${model_types[$i]}-${models[$i]}-${batch_sizes[$i]}-closed-0"
    done
    cmd="./run_expr.sh \
            --device-type ${device_type} \
            --device-id ${device_id} \
            --duration ${duration} \
            --mode mps-uncap \
            --load 1 \
            ${params}"
    echo "Running closed loop experiment:"
    echo "Command used: ${cmd}"
    eval ${cmd}
}

generate_distribution_load() {
    for mode in ${MODES[@]}
    do
        for ((i = $(multiply_and_round ${load_start} 10 0); i <= $(multiply_and_round ${load_end} 10 0); i++)); do
            ratio=$(echo "$load_start + ($i - 1) * $load_step" | bc)
            params=""
            for ((j=0; j<${num_procs}; j++))
            do
                mul=$(multiply_and_round ${ratio} ${rps[$j]})
                params="$params ${model_types[$j]}-${models[$j]}-${batch_sizes[$j]}-${distribution}-${mul}"
            done
            cmd="./run_expr.sh \
                --device-type ${device_type} \
                --device-id ${device_id} \
                --duration ${duration} \
                --mode ${mode} \
                --load ${ratio} \
                ${params}"
            echo "${mode} ${ratio}"
            eval ${cmd} >> print_outs.txt 2>&1
        done
    done
}


get_closed_loop_tput_csv() {
    # Gather metrics and generate RPS values
    result_base=$(IFS=- ; echo "${models[*]}")
    for i in "${!models[@]}"; do
        concatenated_string="${concatenated_string}${models[$i]}-${batch_sizes[$i]}-closed_"
    done
    result_id=${concatenated_string%_}
    tput_csv=results/hetro/${result_base}/${result_id}/tput.csv
}


get_closed_loop_rps() {
    read -r -a tput_csv_string <<< "$(tail -n 1 $tput_csv | \
        awk -v n=${num_procs} -F ',' \
        '{split($0, a); for(i=NF-n+1; i<=NF; i++) printf "%s%s", a[i], (i==NF) ? "" : FS}')"
    IFS=',' read -r -a tput_metrics <<< "$tput_csv_string"
    echo "TPUT: ${tput_metrics[@]}"
    rps=()
    for ((i=0; i<${num_procs}; i++))
    do
       met=$(divide_and_round ${tput_metrics[$i]} ${batch_sizes[$i]})
       rps+=(${met})
    done
    echo "RPS: ${rps[@]}"
}


print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-paramete ..."
    echo "Options:"
    echo "  --device-type   DEVICE_TYPE                   v100, a100, h100                                   (required)"
    echo "  --device-id     DEVICE_ID                     0, 1, 2, ..                                        (required)"
    echo "  --duration      DURATION_OF_EXPR_IN_SECONDS   10                                                 (default 120)"
    echo "  --distribution  DISTRIBUTION_TYPE             poisson, closed, point                             (default poisson)"
    echo "  --load-start    LOAD_START                    0.1                                                (default 0.1)"
    echo "  --load-end      LOAD_END                      1.5                                                (default 1.5)"
    echo "  --load-step     LOAD_STEP                     0.1                                                (default 0.1)"
    echo "  -h, --help                                    Show this help message"
    echo -e "\n"

    echo "Examples:"
    echo " $0 --device-type v100 --device-id 0 --duration 10 vision-vgg19-32"
    echo " $0 --device-type v100 --device-id 1 --duration 20 vision-vgg19-32 vision-mobilenet_v2-1"
    echo " $0 --device-type h100 --device-id 0 --duration 100 vision-vgg19-32 vision-mobilenet_v2-2"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
}


model_run_params=()
duration=120
distribution=poisson
load_start=0.1
load_end=1.5
load_step=0.1
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
            duration=$2
            shift 2
            ;;
        --distribution)
            distribution="$2"
            shift 2
            ;;
        --load-start)
            load_start=$2
            shift 2
            ;;
        --load-end)
            load_end=$2
            shift 2
            ;;
        --load-step)
            load_step=$2
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

num_procs=${#model_run_params[@]}

if [[ -z ${device_type} || (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
    echo "Invalid device_type: ${device_type}"
    print_help
    exit 1
fi

if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
    echo "Invalid device_id: ${device_id}"
    print_help
    exit 1
fi

if [[ ${num_procs} -eq 0 ]]; then
    echo "Need at least 1 model configuration to run"
    print_help
    exit 1
fi

if [[ ${distribution} != "poisson" && ${distribution} != "point" && ${distribution} != "closed" ]]; then
    echo "Invalid distribution: ${distribution}"
    print_help
    exit 1
fi

pattern="^[^-]+-[^-]+-[^-]+$"
model_types=()
models=()
batch_sizes=()
for (( i=0; i<${num_procs}; i++ ))
do
    element=${model_run_params[$i]}
    if [[ ! "${element}" =~ $pattern ]]; then
        echo "Expected: ModelType-Model-BatchSize"
        echo "Got: ${element}"
        echo "Example: vision-densenet121-32"
        exit 1
    fi
    model_types[$i]=$(echo $element | cut -d'-' -f1)
    models[$i]=$(echo $element | cut -d'-' -f2)
    batch_sizes[$i]=$(echo $element | cut -d'-' -f3)
done


# Setup docker for tie-breaker
setup_tie_breaker_container ${device_id}

# Generate closed loop load
generate_closed_loop_load
get_closed_loop_rps

# Generate distribution load
generate_distribution_load
