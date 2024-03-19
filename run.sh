#!/bin/bash -e

log() {
    echo -e "$@"
    echo -e "$@" >> ${PRINT_OUTS}
}

print_log_location() {
    local run_sh_exit_code=$?
    log "Exiting with error_code=$run_sh_exit_code (0 is clean exit)"
    log "Examine ${PRINT_OUTS} for logs"
    exit ${run_sh_exit_code}
}

run_cmd() {
    local cmd_arg="$1"
    local device_id_arg=$2
    local mode_arg=$3
    local duration_arg=$4

    echo "Running: ${cmd_arg}" >> ${PRINT_OUTS}
    eval "${cmd_arg} >> ${PRINT_OUTS} 2>&1 &"
    run_expr_pid=$!

    # Wait for the pipe to be created
    pipe=/tmp/${run_expr_pid}
    local ctr=0
    while [[ ! -p ${pipe} && $ctr -lt 100 ]];
    do
        sleep 0.01
        ctr=$((ctr+1))
    done

    # Start the experiment
    timeout 1 bash -c "echo '{\"device-id\": ${device_id_arg}, \"mode\": \"${mode_arg}\"}' > ${pipe}"

    # Wait for duration_arg + delta
    duration_arg_with_delta=$((duration_arg+5))
    local ctr=0
    while :
    do
        sleep 1
        ctr=$((ctr+1))

        # Waiting for sleep duration
        if [[ ${ctr} -eq ${duration_arg_with_delta} ]]; then
            break
        fi

        # If the experiment dies before that => then error
        if ! taskset -c 0 kill -0 ${run_expr_pid} 2>/dev/null; then
            exit 1
        fi
    done

    # Stop the experiment
    timeout 1 bash -c "echo '{\"mode\": \"stop\"}' > ${pipe}"

    # Wait for the process to exit
    wait ${run_expr_pid}
    run_expr_exit_code=$?
    if [[ ${run_expr_exit_code} -ne 0 ]]; then
        exit ${run_expt_exit_code}
    fi
    echo -e "===============================\n\n" >> ${PRINT_OUTS}
}

generate_json_input() {
    export model_type="\"$1\""
    export model="\"$2\""
    export batch_size=$3
    export distribution_type="\"$4"\"
    export images_per_second=$5
    export slo_percentile=0
    export slo_hw=1000000000
    export slo_lw=1000000000
    export ctrl_grpc=null

    template=$(cat model-param.json.template)
    json_input=$(echo $template | envsubst)

    unset model_type model batch_size distribution_type images_per_second
    unset slo_percentile slo_hw slo_lw ctrl_grpc
}

generate_model_params() {
    declare -a json_arr=("${!1}")
    model_params='[]'
    for json_str in "${json_arr[@]}"; do
        model_params=$(jq ". += [$json_str]" <<< "$model_params")
    done
}

generate_closed_loop_load() {
    json_array=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        generate_json_input ${model_types[$i]} ${models[$i]} ${batch_sizes[$i]} "closed" 0
        json_array+=("${json_input}")
    done

    generate_model_params json_array[@]
    for mode in ${modes[@]}
    do
        cmd="./run_job_mix.sh --device-type ${device_type} --load 1 '${model_params}'"
        log "Running closed loop experiment for ${mode}"
        run_cmd "${cmd}" ${device_id} ${mode} ${duration}
    done
}

generate_distribution_load() {
    get_rps

    for mode in ${modes[@]}
    do
        for ((i = $(multiply_and_round ${load_start} 10 0); i <= $(multiply_and_round ${load_end} 10 0); i++)); do
            ratio=$(echo "$load_start + ($i - 1) * $load_step" | bc)
            json_array=()
            for ((j=0; j<${num_procs}; j++))
            do
                mul=$(multiply_and_round ${ratio} ${rps[$j]})
                echo ${mul}
                generate_json_input ${model_types[$i]} ${models[$i]} ${batch_sizes[$i]} ${distribution} ${mul}
                json_array+=("${json_input}")
            done

            generate_model_params json_array[@]
            cmd="./run_job_mix.sh \
                --device-type ${device_type} \
                --load ${ratio} \
                '${model_params}'"

            log "${mode} ${ratio}"

            run_cmd "${cmd}" ${device_id} ${mode} ${duration}
        done
    done
}

get_rps() {
    closed_loop_result_dir=$(echo "${result_dir}" | sed "s|${distribution}|closed|g")
    tput_csv=${closed_loop_result_dir}/tput.csv

    max_sum=0
    while IFS=, read -r mode load rest; do
        IFS=, read -ra cols <<< "$rest"
        sum=0
        for col in "${cols[@]}"; do
            sum=$(echo "$sum + $col" | bc)
        done
        if (( $(echo "$sum > $max_sum" | bc -l) )); then
            max_sum=$sum
            tput=("${cols[@]}")
        fi
    done < <(tail -n +2 "${tput_csv}")

    if [[ ${#tput[@]} -ne ${num_procs} ]]; then
        log "Unable to get throughput metrics for all models"
        exit 1
    fi
    log "TPUTS: ${tput[@]}"

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
    PRINT_OUTS=/tmp/print_outs-$(uuidgen | cut -c 1-8).txt
    rm -f ${PRINT_OUTS}
    echo "Logs at ${PRINT_OUTS}"
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
