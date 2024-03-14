#!/bin/bash -e

print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-parameter ..."
    echo "Options:"
    echo "  --device-type DEVICE_TYPE                   v100, a100, h100                                   (required)"
    echo "  --load        LOAD_INDICATOR                0.5, 0.2, 1                                        (default 1)"
    echo "  -h, --help                                  Show this help message"
    echo -e "\n"

    echo "NOTE:"
    echo "  Load generation is done via Distribution Type and RPS"
    echo -e "\n"

    echo "Examples:"
    echo " $0 --device-type v100 --load 1 vision-vgg19-32-point-10"
    echo " $0 --device-type v100 --load 0.8 vision-vgg19-32-poisson-10 vision-mobilenet_v2-1-closed-100"
    echo " $0 --device-type h100 --load 0.4 vision-vgg19-32-point-10 vision-mobilenet_v2-1-point-10"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
}

get_input() {
    # Parse arguments
    model_run_params=()
    load=1
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                device_type="$2"
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
    num_procs=${#model_run_params[@]}
}

modes_ran=()
device_ids_ran=()
uuids_ran=()
cleanup_handler() {
    exit_code=$?

    # Kill any pending procs
    IFS="|"
    uuid_grep="${uuids_ran[*]}"
    unset IFS
    ps -eaf | grep batched_inference_executor.py | grep ${uuid_grep} |
        grep -v grep | grep -v docker | awk '{print $2}' |
        xargs -I{} kill -9 {} || :

    # Clean the modes ran
    for ((i=0; i<${#modes_ran[@]}; i++))
    do
        cleanup ${modes_ran[$i]} ${device_ids_ran[$i]}
    done

    # Clean up the fifo pipe created
    rm -f ${fifo_pipe} || :

    # Clean up the IPC queue
    ipcrm --all=msg

    # Exit with the exit code with which the handler was called
    exit ${exit_code}
}


validate_input() {
    if [[ (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
        echo "Invalid device_type: ${device_type}"
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
}

read_queue() {
    local qid=$1
    if [[ ${USE_DOCKER} -eq 1 ]]; then
        setup_tie_breaker_container
        docker_prefix="${DOCKER} exec -it ${tie_breaker_ctr}"
    fi

    cmd="${docker_prefix} python3 -c \
        \"from src.utils import SysVQueue; \
        from queue import Queue; \
        SysVQueue(${qid}, create_new_queue=False).read_queue(Queue())\""

    eval ${cmd} &
}


parse_input() {
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


    echo "num_procs=${num_procs}"
    echo "model_types=${model_types[@]}"
    echo "models=${models[@]}"
    echo "batch_sizes=${batch_sizes[@]}"
    echo "distribution_types=${distribution_types[@]}"
    echo "rps=${rps[@]}"
}

run_orion_expr() {
    local mode_arg=$1
    local device_id_arg=$2
    local run_uuid_arg=$3

    # Setup directory to collect stats
    tmpdir=$(mktemp -d)

    # Setup orion container
    setup_orion_container ${device_id_arg} ${run_uuid_arg}

    # Run the experiment
    # TODO: Fix duration
    model_defn_string="${model_run_params[*]}"
    ${DOCKER} exec -it ${ORION_CTR} bash -c \
        "LD_PRELOAD='/root/orion/src/cuda_capture/libinttemp.so' \
        python3.8 src/orion_scheduler.py \
        --device-type ${device_type} \
        --duration ${duration} \
        ${model_defn_string}"

    # Copy the results
    rm -f ${tmpdir}/*
    ${DOCKER} exec ${ORION_CTR} sh -c "ls /tmp/*.pkl" | while read -r file; do
        ${DOCKER} cp orion:${file} ${tmpdir}
    done

    # Collect the pickle files
    pkl_files=()
    for f in $(ls ${tmpdir})
    do
        pkl_files+=(${tmpdir}/${f})
    done

    compute_stats pkl_files[@] ${mode_arg} ${load} ${result_dir}
}

run_other_expr() {
    local mode_arg=$1
    local device_id_arg=$2
    local run_uuid_arg=$3
    declare -a prev_proc_arg=("${!4}")
    if [[ ${#prev_proc_arg[@]} -gt 0 ]]; then
        is_prev=1
    fi

    assert_mig_status ${mode_arg} ${device_id_arg}
    enable_mps_if_needed ${mode_arg} ${device_id_arg}
    setup_mig_if_needed ${mode_arg} ${device_id_arg} ${num_procs}

    if [[ ${mode_arg} == "mps-miglike" ]]; then
        percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode_arg} == "mps-equi" ]]; then
        percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
    fi

    cci_uuid=($(nvidia-smi -L | grep MIG- | awk '{print $6}' | sed 's/)//g'))
    chunk_id=($(nvidia-smi -L | grep MIG- | awk '{print $2}' | sed 's/)//g'))
    cmd_arr=()

    if [[ ${USE_DOCKER} == 1 ]]; then
        if [[ ${mode_arg} == "mig" ]]; then
            printf -v devices "%s," "${cci_uuid[@]}"
            devices=${devices%,}
            setup_tie_breaker_container ${devices} ${run_uuid_arg}
        else
            setup_tie_breaker_container ${device_id_arg} ${run_uuid_arg}
        fi
    fi

    if [[ ${USE_DOCKER} -eq 1 ]]; then
        # Assume only 1 device per container
        docker_prefix="${DOCKER} exec -d -it ${tie_breaker_ctr} bash -c '"
    fi

    # In MPS mode_arg, we knowingly set CUDA_VISIBLE_DEVICES to 0 (not a code bug)
    # Reasoning:
    # * When CUDA_VISIBLE_DEVICES is set before launching the control daemon,
    #   the devices will be remapped by the MPS server.
    # * This means that if your system has devices 0, 1 and 2,
    #   and if CUDA_VISIBLE_DEVICES is set to "0,2", then when a client
    #   connects to the server it will see the remapped devices: device 0
    #   and a device 1.
    # * Therefore, keeping CUDA_VISIBLE_DEVICES set to "0,2"
    #   when launching the client would lead to an error.
    # Ref: https://docs.nvidia.com/deploy/mps/index.html#topic_5_2
    for (( c=0; c<${num_procs}; c++ ))
    do
        if [[ ${mode_arg} == "mig" ]]; then
            echo "Setting ${chunk_id[$c]} for ${models[$c]}"
            export_prefix="export CUDA_VISIBLE_DEVICES=${cci_uuid[$c]}"
        elif [[ ${mode_arg} == "mps-miglike" || ${mode_arg} == "mps-equi" ]]; then
            echo "Setting ${percent[$c]}% for ${models[$c]}"
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1 && \
                           export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${percent[$c]} && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0"
        elif [[ ${mode_arg} == "mps-uncap" ]]; then
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0 && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0"
        else
            export_prefix="export CUDA_VISIBLE_DEVICES=${device_id_arg}"
        fi


        cmd="${docker_prefix} ${export_prefix} && \
            python3 src/batched_inference_executor.py \
            --model-type ${model_types[$c]} \
            --model ${models[$c]} \
            --batch-size ${batch_sizes[$c]} \
            --distribution_type ${distribution_types[$c]} \
            --rps ${rps[$c]} \
            --tid ${c}
            --uid ${run_uuid_arg}"

        if [[ ${is_prev} -eq 1 ]]; then
            cmd="${cmd} --qid ${prev_proc_arg[$c]}"
        fi

        if [[ ${USE_DOCKER} -eq 1 ]]; then
            cmd+="'"
        else
            cmd+=" > /dev/null &"
        fi

        eval $cmd
        cmd_arr+=("${cmd}")
    done

    readarray -t forked_pids < <(ps -eaf | grep batched_inference_executor.py |
        grep ${run_uuid_arg} | grep -v grep | grep -v docker |
        awk '{for (i=1; i<=NF; i++) if ($i == "--tid") print $(i+1),$0}' |
        sort -n | cut -d' ' -f2- | awk '{print $2}')
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
            sleep 0.25
        done

        if [[ ${load_ctr} -eq 0 ]]; then
            echo "Some of the models did not load!"
            echo "Examine commands: "
            for cmd in "${cmd_arr[@]}"
            do
                echo "  ${cmd}"
            done
            exit 1
        fi
    done

    # Start inference
    echo "Starting inference on ${loaded_procs[@]}"
    kill -SIGUSR1 ${loaded_procs[@]}
}

read_fifo()
{
    pipe_name=$1
    read json_data < ${pipe_name}
    mode_to_run=$(echo "$json_data" | jq -r '.mode')
    device_id_to_run=$(echo "$json_data" | jq -r '.device_id')
    modes_ran+=(${mode_to_run})
    device_ids_ran+=(${device_id_to_run})
}

run_expr()
{
    # Procs run so far
    local procs=()
    local prev=()

    # Number of modes run
    num_modes_run=0

    read_fifo ${fifo_pipe}
    while [[ ${mode_to_run} != "stop" ]];
    do
        # Enable PM in GPU
        ${SUDO} nvidia-smi -i ${device_id_to_run} -pm ENABLED

        # acquire lock on the GPU
        lock_gpu ${device_id_to_run}

        # Get a unique id for the run
        local run_uuid=$(uuidgen)
        uuids_ran+=(${run_uuid})

        # Start the experiment
        if [[ ${mode_to_run} == "orion" ]]; then
            run_orion_expr ${mode_to_run} ${device_id_to_run} ${run_uuid}
        else
            run_other_expr ${mode_to_run} ${device_id_to_run} ${run_uuid} prev[@]
        fi

        # Make the previously started processes to stop
        if [[ ${#prev[@]} -gt 0 ]]; then
            kill -SIGUSR2 ${prev[@]}
        fi

        # increment the number of runs
        num_modes_run=$((num_modes_run+1))
        mode_run=${mode_to_run}

        # Make current => previous
        prev=("${loaded_procs[@]}")
        procs+=( "${loaded_procs[@]}" )

        prev_device_id_run=${device_id_to_run}
        read_fifo ${fifo_pipe}
        unlock_gpu ${prev_device_id_run}
    done

    # Make sure to stop all inferences
    if [[ ${#prev[@]} -gt 0 ]]; then
        kill -SIGUSR2 ${prev[@]}
        for prev_pid in ${prev[@]}
        do
            read_queue ${prev_pid}
        done
    fi

    # Clean the created pipe
    rm -f ${mode_arg_pipe}

    # Update the mode_run
    if [[ ${num_modes_run} -gt 1 ]]; then
        mode_run="tie-breaker"
    fi

    # Get stats
    pkl_files=()
    for pid in "${procs[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
        pkl_file="/tmp/${pid}.pkl"
        if [[ -f ${pkl_file} ]]; then
            pkl_files+=(${pkl_file})
        fi
    done

    if [[ ${#pkl_files[@]} -gt 0 ]]; then
        compute_stats pkl_files[@] ${mode_run} ${load} ${result_dir}
    else
        echo "No modes were run, so not computing stats"
    fi
}

compute_stats()
{
    declare -a pkl_files_arg=("${!1}")
    local mode_arg=$2
    local load_arg=$3
    local result_dir_arg=$4

    if [[ ${USE_DOCKER} -eq 1 ]]; then
        setup_tie_breaker_container
        docker_prefix="${DOCKER} exec -it ${tie_breaker_ctr}"
    fi

    cmd="${docker_prefix} python3 src/stats.py \
        --mode ${mode_arg} \
        --load ${load_arg} \
        --result_dir ${result_dir_arg} \
        ${pkl_files_arg[@]}"
    eval ${cmd}

    echo "Results stored in: ${result_dir_arg}"
}

setup_expr()
{
    # Source helper
    source helper.sh && helper_setup
    trap cleanup_handler EXIT

    # Find where to store results
    get_result_dir models[@] batch_sizes[@] distribution_types[@] ${device_type}
    mkdir -p ${result_dir}

    # Create a FIFO to listen on
    fifo_pipe=/tmp/$$
    rm -f ${fifo_pipe}
    mkfifo ${fifo_pipe}

    # Clean up the IPC queue
    ipcrm --all=msg
}

get_input $@
validate_input
parse_input
setup_expr
run_expr
