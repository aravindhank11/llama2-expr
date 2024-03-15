#!/bin/bash

DOCKER="docker"

# tie breaker details
TIE_BREAKER_CTR_PREFIX="tie-breaker"
TIE_BREAKER_IMG="aravindhank11/tie-breaker"

# orion details
ORION_CTR_PREFIX="orion"
ORION_IMG="fotstrt/orion-ae:v1"
ORION_FORK="orion-fork"

if [[ $USE_SUDO == 1 ]]; then
    SUDO="sudo"
    DOCKER="sudo docker"
fi

function helper_setup() {
    if [[ ${USE_DOCKER} == 1 && ! -z ${VENV} ]]; then
        echo "Set ONLY one of USE_DOCKER or VENV env variables and not both"
        exit 1
    fi

    if [[ ${USE_DOCKER} -ne 1 && -z ${VENV} ]]; then
        echo "Set EXACTLY one of USE_DOCKER or VENV env variables"
        exit 1
    fi

    if [[ ! -z ${VENV} ]]; then
        source ${VENV}/bin/activate
    fi
}

function is_mig_feature_available() {
    echo $(nvidia-smi --query-gpu=name --format=csv,noheader | egrep -i "a100|h100" | wc -l)
}

function assert_mig_status()
{
    mode=$1
    device_id=$2
    if [[ ${mode} == "mig" ]]; then
        if [[ $(${SUDO} nvidia-smi -i ${device_id} --query-gpu=mig.mode.current --format=csv | grep "Enabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode not enabled"
            exit 1
        fi
    else
	mig_gpu=$(is_mig_feature_available)
        if [[ $mig_gpu -ne 0 && $(${SUDO} nvidia-smi -i ${device_id} --query-gpu=mig.mode.current --format=csv | grep "Disabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode is not disabled"
            exit 1
        fi
    fi
}

function enable_mps_if_needed()
{
    local mode=$1
    local device_id=$2
    if [[ ${mode} == mps-* ]]; then
        echo "Enabling MPS"
        export CUDA_VISIBLE_DEVICES=${device_id}
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id}
        export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${device_id}
        ${SUDO} nvidia-smi -i ${device_id} -c EXCLUSIVE_PROCESS
        nvidia-cuda-mps-control -d
        unset CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 1 ]]; then
            echo "Unable to enable MPS"
            exit 1
        fi
    fi
}

function disable_mps_if_needed()
{
    local mode=$1
    local device_id=$2
    if [[ ${mode} == mps-* ]]; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id}
        echo quit | nvidia-cuda-mps-control
        ${SUDO} nvidia-smi -i ${device_id} -c DEFAULT
        ${SUDO} rm -fr /tmp/mps_${device_id} || :
        ${SUDO} rm -fr /tmp/mps_log_$i || :

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 0 ]]; then
            echo "Unable to disable MPS"
            exit 1
        fi
    fi
}

function setup_mig_if_needed()
{
    local mode=$1
    local device_id=$2
    local num_procs=$3
    if [[ ${mode} != "mig" ]]; then
        return
    fi

    if [[ $# -eq 3 && $3 == "chunks" ]]; then
        gi=("" "19" "19,19" "19,19,19" "19,19,19,19" "19,19,19,19,19" "19,19,19,19,19,19" "19,19,19,19,19,19,19")
    else
        gi=("" "0" "9,5" "9,14,14" "14,14,14,19" "14,14,19,19,19" "14,19,19,19,19,19" "19,19,19,19,19,19,19")
    fi


    # Create Gpu Instance
    ${SUDO} nvidia-smi mig -i ${device_id} -cgi "${gi[${num_procs}]}"

    # Get GPU Instance ID
    gi_id_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lgi | awk '{print $6}' | grep -P '^\d+$'))

    # Create Compute Instance
    for gi_id in "${gi_id_arr[@]}"
    do
        # Choose the largest sub-chunk
        ci=$(${SUDO} nvidia-smi mig -i ${device_id} -gi ${gi_id} -lcip | grep "\*" | awk '{print $6}' | tr -d "*")
        ${SUDO} nvidia-smi mig -i ${device_id} -gi ${gi_id} -cci ${ci}
    done
}

function cleanup_mig_if_needed()
{
    mode=$1
    device_id=$2
    if [[ ${mode} != "mig" ]]; then
        return
    fi

	# Delete the GPU profile
	exit_code=0
	while :;
	do
        # Cleanup Compute Instances
        ci_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lci | awk '{print $7}' | grep -P '^\d+$'))
        gi_arr=($(${SUDO} nvidia-smi mig -i ${device_id} -lci | awk '{print $3}' | grep -P '^\d+$'))
        i=0
        while [[ $i -lt ${#gi_arr[*]} ]];
        do
            ci=${ci_arr[$i]}
            gi=${gi_arr[$i]}
            ${SUDO} nvidia-smi mig -i ${device_id} -dci -ci ${ci} -gi ${gi}
            i=$(( $i + 1))
        done

        # Cleanup GPU Instances
        ${SUDO} nvidia-smi mig -i ${device_id} -dgi
        exit_code=$?
        if [[ ${exit_code} -eq 0 || ${exit_code} -eq 6 ]]; then
            break
        fi
        echo "${SUDO} nvidia-smi mig -i ${device_id} -dgi failed. Trying in 1s"
        sleep 1
    done
}

function lock_gpu() {
    local device_id=$1
    echo "Attempting to acquire exclusive lock for GPU: ${device_id}"
    exec 9>/tmp/gpu_${device_id}.lock
    flock -x 9
    echo "Acquired exclusive lock for GPU: ${device_id}"
}

function unlock_gpu() {
    local device_id=$1
    exec 9>&-
}

function cleanup()
{
    mode=$1
    device_id=$2
    disable_mps_if_needed ${mode} ${device_id}
    cleanup_mig_if_needed ${mode} ${device_id}
    cleanup_orion_container
    cleanup_tie_breaker_containers
    unlock_gpu ${device_id}
}

function calc_mean_sd() {
    arr=("$@")
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    sd=$(echo ${arr[@]} | awk -vM=$mean '{for(i=1;i<=NF;i++){sum+=($i-M)*($i-M)};print sqrt(sum/NF)}')
    echo "$mean, $sd"
}

function calc_min_max_mean {
    arr=("$@")
    max=$(echo ${arr[@]} | awk -vmax=${arr[0]} '{for(i=1;i<=NF;i++){(( $i > max )) && max=$i};print max}')
    min=$(echo ${arr[@]} | awk -vmin=${arr[0]} '{for(i=1;i<=NF;i++){(( $i < min )) && min=$i};print min}')
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    echo "$min, $max, $mean"
}

function wait_till_one_process_exits {
    pids=("$@")
    cpu=0
    done=0
    while true; do
        for pid in ${pids[@]}; do
            if ! taskset -c ${cpu} kill -0 $pid >/dev/null 2>&1; then
                done=1
                break
            fi
        done

        if [[ ${done} -eq 1 ]]; then
            echo "Sending SIGUSR2 to ${pids[@]}"
            kill -SIGUSR2 ${pids[@]} >/dev/null 2>&1
            break
        fi
	done

    for pid in ${pids[@]}; do
        echo "Waiting for ${pid} to finish"
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    done
}

function setup_orion_container {
    local device_id=$1
    local uuid=$2
    local WS=$(git rev-parse --show-toplevel)
    local DOCKER_WS=/root/$(basename ${WS})

    if [[ -z ${uuid} ]]; then
        uuid=$(uuidgen)
    fi

    # Setup container
    cleanup_orion_container
    orion_ctr=${ORION_PREFIX}"-"${uuid}
    ${DOCKER} run -v ${WS}:${DOCKER_WS} -it -d \
        -w ${DOCKER_WS} \
        --name ${orion_ctr} \
        --ipc=host --pid=host \
        --gpus "device=${device_id}" \
        ${ORION_IMG} bash > /dev/null

    # Install necessary package
    NSIGHT_COMPUTE_TAR=nsight-compute.tar
    if [[ ! -f ${NSIGHT_COMPUTE_TAR} ]]; then
        # pip install gdown
        cmd="gdown --id 1_HY1FOIS6KP7dLTKRZ30Wliqc9P_N7hu"
        exit_code=0
        eval ${cmd} || exit_code=1
        if [[ ${exit_code} -ne 0 ]]; then
            echo "gdown package not found!"
            echo "Install using: 'pip install gdown'"
            echo "Or try running command: '${cmd}', fix it and re-call the script"
            exit 1
        fi
    fi
    ${DOCKER} cp ${NSIGHT_COMPUTE_TAR} ${orion_ctr}:/usr/local/ > /dev/null 2>&1
    ${DOCKER} exec -it ${orion_ctr} bash -c "tar -xf /usr/local/nsight-compute.tar -C /usr/local/ > /dev/null 2>&1"
    ${DOCKER} exec -it ${orion_ctr} bash -c "wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1"
    ${DOCKER} exec -it ${orion_ctr} bash -c "dpkg -i nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1 && rm -f nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb"
    ${DOCKER} exec -it ${orion_ctr} bash -c "pip install transformers > /dev/null 2>&1"
}

function setup_tie_breaker_container {
    local devices=$1
    local uuid=$2
    local WS=$(git rev-parse --show-toplevel)
    local DOCKER_WS=/home/$USER/$(basename ${WS})

    if [[ -z ${devices} ]]; then
        devices="all"
    fi

    if [[ -z ${uuid} ]]; then
        uuid=$(uuidgen)
    fi

    # Setup container
    cleanup_tie_breaker_containers
    tie_breaker_ctr=${TIE_BREAKER_CTR_PREFIX}"-"${uuid}
    cmd="${DOCKER} run -v ${WS}:${DOCKER_WS} -it -d \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /tmp:/tmp \
        -w ${DOCKER_WS} \
        -u $(id -u $USER):$(id -g $USER) \
        --name ${tie_breaker_ctr} \
        --ipc=host --pid=host \
        --gpus '\"device=${devices}\"' \
        ${TIE_BREAKER_IMG} bash > /dev/null"
    echo ${cmd}
    eval ${cmd}
}

function cleanup_orion_container {
    ${DOCKER} ps -a | grep ${ORION_CTR_PREFIX} | awk '{print $NF}' | \
        xargs -I{} ${DOCKER} rm -f {} >/dev/null 2>&1 || :
}

function cleanup_tie_breaker_containers {
    ${DOCKER} ps -a | grep ${TIE_BREAKER_CTR_PREFIX} | awk '{print $NF}' | \
        xargs -I{} ${DOCKER} rm -f {} >/dev/null 2>&1 || :
}

function multiply_and_round() {
    local result=$(echo "$1 * $2" | bc)
    if [[ $# -eq 3 ]]; then
        printf "%.$3f\n" "$result"
    else
        printf "%.2f\n" "$result"
    fi
}

function divide_and_round() {
    local result=$(echo "scale=2; $1 / $2" | bc)
    printf "%.2f\n" "$result"
}

function get_result_dir() {
    declare -a models_arr=("${!1}")
    declare -a batch_sizes_arr=("${!2}")
    declare -a distribution_types_arr=("${!3}")
    local device_type_arg=$4


    local result_base=$(IFS=- ; echo "${models_arr[*]}")
    for i in "${!models_arr[@]}"; do
        concatenated_string="${concatenated_string}${models_arr[$i]}-${batch_sizes_arr[$i]}-${distribution_types_arr[$i]}_"
    done
    local result_id=${concatenated_string%_}
    result_dir=results/${device_type_arg}-1/${result_base}/${result_id}
}


mps_mig_percentages=("" "100" "57,43" "42,29,29" "29,29,28,14" "29,29,14,14,14" "29,15,14,14,14,14" "15,15,14,14,14,14,14")
mps_equi_percentages=("" "100" "50,50" "34,33,33" "25,25,25,25" "20,20,20,20,20" "17,17,17,17,16,16" "15,15,14,14,14,14,14")
# Command to profile
# nsys profile --stats=true --force-overwrite true --wait=all -o trial
