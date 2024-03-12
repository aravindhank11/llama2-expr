#!/bin/bash -e

PRINT_OUTS=print_outs.txt
log() {
    echo -e "$@"
    echo -e "$@" >> ${PRINT_OUTS}
}

print_log_location() {
    exit_code=$?
    log "Exiting with error_code=$exit_code (0 is clean exit)"
    log "Examine ${PRINT_OUTS} for logs"
    exit ${exit_code}
}

generate_closed_loop_load() {
    for (( i=0; i<${num_procs}; i++ ))
    do
        params="$params ${model_types[$i]}-${models[$i]}-${batch_sizes[$i]}-closed-0"
    done

    for mode in ${modes[@]}
    do
        cmd="./run_expr.sh \
                --device-type ${device_type} \
                --device-id ${device_id} \
                --duration ${duration} \
                --mode ${mode} \
                --load 1 \
                ${params}"

        log "Running closed loop experiment for ${mode}:"
        echo "Running: ${cmd}" >> ${PRINT_OUTS}
        eval ${cmd} >> ${PRINT_OUTS} 2>&1
        echo -e "===============================\n\n" >> ${PRINT_OUTS}
    done
}

generate_distribution_load() {
    get_rps

    for mode in ${modes[@]}
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
            log "${mode} ${ratio}"
            echo "Running: ${cmd}" >> ${PRINT_OUTS}
            eval ${cmd} >> ${PRINT_OUTS} 2>&1
            echo -e "===============================\n\n" >> ${PRINT_OUTS}
        done
    done
}

get_rps() {
    closed_loop_result_dir=$(echo "${result_dir}" | sed "s|${distribution}|closed|g")
    tput_csv=${closed_loop_result_dir}/tput.csv

    tput=()
    for ((i = 3; i <= 3 + ${num_procs}; i++));
    do
        echo "tail -n+2 ${tput_csv} | cut -d ',' -f${i} | sort -n | tail -1"
        max_value=$(tail -n+2 ${tput_csv} | cut -d ',' -f${i} | sort -n | tail -1)
        tput+=(${max_value})
    done

    if [[ ${#tput[@]} -ne ${num_procs} ]]; then
        log "Unable to get throughput metrics for all models"
        exit 1
    fi
    log "TPUTS: ${tput[@]}"

    rps=()
    for ((i=0; i<${num_procs}; i++))
    do
       met=$(divide_and_round ${tput[$i]} ${batch_sizes[$i]})
       rps+=(${met})
    done
    log "RPS: ${rps[@]}"
}


print_help() {
    log "Usage: ${0} [OPTIONS] model1-parameter model2-paramete ..."
    log "Options:"
    log "  --device-type   DEVICE_TYPE                   v100, a100, h100                                   (required)"
    log "  --device-id     DEVICE_ID                     0, 1, 2, ..                                        (required)"
    log "  --modes         MODE1,MODE2,MODE3             mps-uncap,orion,ts,mig                             (default mps-uncap,ts)"
    log "  --duration      DURATION_OF_EXPR_IN_SECONDS   10                                                 (default 120)"
    log "  --distribution  DISTRIBUTION_TYPE             poisson, closed, point                             (default poisson)"
    log "  --load-start    LOAD_START                    0.1                                                (default 0.1)"
    log "  --load-end      LOAD_END                      1.5                                                (default 1.5)"
    log "  --load-step     LOAD_STEP                     0.1                                                (default 0.1)"
    log "  -h, --help                                    Show this help message"
    log -e "\n"

    log "Examples:"
    log " $0 --device-type v100 --device-id 0 --duration 10 vision-vgg19-32"
    log " $0 --device-type v100 --device-id 1 --duration 20 vision-vgg19-32 vision-mobilenet_v2-1"
    log " $0 --device-type h100 --device-id 0 --duration 100 vision-vgg19-32 vision-mobilenet_v2-2"
    log -e "\n"

    log "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    log "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    log "--> reboot"
}

get_input() {
    model_run_params=()
    duration=120
    distribution=poisson
    load_start=0.1
    load_end=1.5
    load_step=0.1
    modes=("mps-uncap" "ts")
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
            --modes)
                unset modes
                IFS=',' read -r -a modes <<< "$2"
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
}

validate_input() {
    num_procs=${#model_run_params[@]}
    if [[ -z ${device_type} || (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
        log "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        log "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    if [[ ${num_procs} -eq 0 ]]; then
        log "Need at least 1 model configuration to run"
        print_help
        exit 1
    fi

    if [[ ${distribution} != "poisson" && ${distribution} != "point" && ${distribution} != "closed" ]]; then
        log "Invalid distribution: ${distribution}"
        print_help
        exit 1
    fi

    for mode in ${modes[@]}
    do
        if [[ ${mode} != "mps-uncap" && ${mode} != "mps-equi" && ${mode} != "mps-miglike" && ${mode} != "mig" && ${mode} != "ts" && ${mode} != "orion" ]]; then
            log "Invalid mode: ${mode}"
            log "Must be one of: mps-uncap, mps-equi, mps-miglike, mig, ts, orion"
            print_help
            exit 1
        fi
    done
}

parse_input() {
    pattern="^[^-]+-[^-]+-[^-]+$"
    model_types=()
    models=()
    models_and_batch_sizes=()
    batch_sizes=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        element=${model_run_params[$i]}
        if [[ ! "${element}" =~ $pattern ]]; then
            log "Expected: ModelType-Model-BatchSize"
            log "Got: ${element}"
            log "Example: vision-densenet121-32"
            exit 1
        fi
        model_types[$i]=$(echo $element | cut -d'-' -f1)
        models[$i]=$(echo $element | cut -d'-' -f2)
        batch_sizes[$i]=$(echo $element | cut -d'-' -f3)
        models_and_batch_sizes[$i]=${models[$i]}"-"${batch_sizes[$i]}
    done
}


setup_expr() {
    source helper.sh && helper_setup
    WS=$(git rev-parse --show-toplevel)
    DOCKER_WS=~/$(basename ${WS})
    distribution_types=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        distribution_types+=(${distribution})
    done
    get_result_dir models[@] batch_sizes[@] distribution_types[@] ${device_type}
    trap print_log_location EXIT
    rm -f ${PRINT_OUTS}
}


get_input $@
validate_input
parse_input
setup_expr
if [[ ${distribution} == "closed" ]]; then
    # Generate closed loop load
    generate_closed_loop_load
else
    # Generate distribution load
    generate_distribution_load
fi
log "Run success: results are stored in ${result_dir}"
