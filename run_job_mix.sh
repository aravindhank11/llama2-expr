#!/bin/bash -e

print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-parameter ..."
    echo "Options:"
    echo "  --device-type   DEVICE_TYPE                   v100, a100, h100                                   (required)"
    echo "  --load          LOAD_INDICATOR                0.5, 0.2, 1                                        (default 1)"
    echo "  --tie-breaker                                                                                    (pass flag for enabling tie-breaker)"
    echo "  -h, --help                                    Show this help message"
    echo -e "\n"

    echo "NOTE:"
    echo "  Load generation is done via Distribution Type and RPS"
    echo -e "\n"

    echo "Example:"
    echo " $0 --device-type a100 --load 1 '[{\"model-type\": \"vision\", \"model\": \"vgg19\", \"batch-size\": 32, \"distribution-type\": \"poisson\", \"rps\": 40}, {\"model-type\": \"vision\", \"model\": \"mobilenet_v2\", \"batch-size\": 4, \"distribution-type\": \"closed\", \"rps\": 0}]'"
    echo ""
    echo " $0 --device-type a100 --load 0.2 --tie-breaker '[{\"model-type\": \"vision\", \"model\": \"vgg19\", \"batch-size\": 32, \"distribution-type\": \"poisson\", \"rps\": 40}, {\"model-type\": \"vision\", \"model\": \"mobilenet_v2\", \"batch-size\": 4, \"distribution-type\": \"point\", \"rps\": 220}]'"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
}

get_input() {
    # Parse arguments
    load=1
    is_tie_breaker=false
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
            --tie-breaker)
                is_tie_breaker=true
                shift 1
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                parse_model_parameters "$@"
                shift $#
            ;;
        esac
    done
    num_procs=${#model_run_params[@]}
}

modes_ran=()
device_ids_ran=()
uuids_ran=()
cleanup_handler() {
    original_x=$(set +o | grep xtrace)
    set -x
    local job_mix_exit_code=$?

    # Kill any pending procs
    if [[ ${#uuids_ran[@]} -gt 0 ]]; then
        IFS="|"
        uuid_grep="${uuids_ran[*]}"
        unset IFS
        ps -eaf | grep batched_inference_executor.py | egrep "${uuid_grep}" |
            grep -v grep | awk '{print $2}' |
            xargs -I{} kill -9 {} || :
    fi

    # cleanup the containers created
    cleanup_orion_containers uuid_grep || :

    # Clean the modes ran
    for ((y=0; y<${#modes_ran[@]}; y++))
    do
        echo "Cleaning up for ${modes_ran[$y]} on ${device_ids_ran[$y]}"
        cleanup ${modes_ran[$y]} ${device_ids_ran[$y]} || :
    done

    # Clean up the fifo pipe created
    rm -f ${fifo_pipe} || :

    # Clean up the IPC queue
    ipcrm --all=msg || :

    # Exit with the exit code with which the handler was called
    echo "Exiting with Error code: ${job_mix_exit_code} (0 is clean exit)"
    eval "$original_x"
    exit ${job_mix_exit_code}
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

    (timeout 5 python3 -c \
        "from src.utils import SysVQueue; \
        from queue import Queue; \
        SysVQueue(${qid}, create_new_queue=False).read_queue(Queue())" || :) &
    read_queue_pids+=($!)
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

    mig_info=$(awk "/^GPU ${device_id_arg}:/{p=1; next} /^GPU/{p=0} p && /^  MIG/{print}" <<< "$(nvidia-smi -L)")
    chunk_id=($(echo "$mig_info" | awk '{print $2}'))
    cci_uuid=($(echo "$mig_info" | awk '{print $NF}' | sed 's/[()]//g'))
    cmd_arr=()

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


        # Assumes: we can run 7 models in parallel in a device
        cpu=$(((device_id_arg * 7) + (c+1)))

        cmd="${export_prefix} && \
            taskset -c ${cpu} python3 src/batched_inference_executor.py \
            ${model_run_params[$c]} \
            --tid ${c}
            --uid ${run_uuid_arg}"
        if [[ ${is_prev} -eq 1 ]]; then
            cmd="${cmd} --qid ${prev_proc_arg[$c]}"
        fi
        cmd+=" > /dev/null &"

        eval $cmd
        cmd_arr+=("${cmd}")
    done

    readarray -t forked_pids < <(ps -eaf | grep batched_inference_executor.py |
        grep ${run_uuid_arg} | grep -v grep |
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
    echo "Got: ${json_data}"
    mode_to_run=$(echo "$json_data" | jq -r '.mode')
    device_id_to_run=$(echo "$json_data" | jq -r '.["device-id"]')
    modes_ran+=(${mode_to_run})
    device_ids_ran+=(${device_id_to_run})
}

run_expr()
{
    # Procs run so far
    local procs=()
    local prev=()

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
        safe_clean_gpu prev[@] ${prev_mode_run} ${prev_device_id_run}

        # Make current => previous
        prev=("${loaded_procs[@]}")
        procs+=( "${loaded_procs[@]}" )

        # Keep track of previous state
        prev_mode_run=${mode_to_run}
        prev_device_id_run=${device_id_to_run}
        read_fifo ${fifo_pipe}
    done

    # Read the queue that is being dumped by the last procs
    read_queue_pids=()
    for prev_pid in ${prev[@]}
    do
        read_queue ${prev_pid}
    done

    # Make sure to stop all inferences
    safe_clean_gpu prev[@] ${prev_mode_run} ${prev_device_id_run}

    # Clean the created pipe
    rm -f ${mode_arg_pipe}

    # Wait for the read queue_pid to exit
    for read_queue_pid in ${read_queue_pids[@]}
    do
        echo "Waiting for read-pid ${read_queue_pid} to exit..."
        wait ${read_queue_pid}
    done

    # Get stats
    pkl_files=()
    for pid in "${procs[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
        pkl_file="/tmp/${pid}.pkl"
        pkl_files+=(${pkl_file})
    done

    mode_run=${prev_mode_run}
    if [[ ${is_tie_breaker} == true ]]; then
        mode_run="tie-breaker"
    fi
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

    python3 src/stats.py \
        --mode ${mode_arg} \
        --load ${load_arg} \
        --result_dir ${result_dir_arg} \
        ${pkl_files_arg[@]}

    echo "Results stored in: ${result_dir_arg}"
}

setup_expr()
{
    trap cleanup_handler EXIT

    # Find where to store results
    get_result_dir models[@] batch_sizes[@] distribution_types[@] ${device_type}
    mkdir -p ${result_dir}

    # Create a FIFO to listen on
    fifo_pipe=/tmp/$$
    rm -f ${fifo_pipe}
    mkfifo ${fifo_pipe}

    # Clean up the IPC queue
    ipcrm --all=msg || :
}

source helper.sh && helper_setup
get_input $@
validate_input
setup_expr
run_expr
