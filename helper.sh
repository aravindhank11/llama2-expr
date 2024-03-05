#!/bin/bash

DOCKER="docker"

# tie breaker details
TIE_BREAKER_CTR="tie-breaker"
TIE_BREAKER_IMG="aravindhank11/tie-breaker"

# orion details
ORION_CTR="orion"
ORION_IMG="fotstrt/orion-ae:v1"
ORION_FORK="orion-fork"

if [[ $USE_SUDO == 1 ]]; then
    SUDO="sudo"
    DOCKER="sudo docker"
fi

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
        ${SUDO} nvidia-smi -i ${device_id} -c EXCLUSIVE_PROCESS
        nvidia-cuda-mps-control -d

        # TODO: Needs fixing
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
        echo quit | nvidia-cuda-mps-control
        ${SUDO} nvidia-smi -i ${device_id} -c DEFAULT

        # TODO: Needs fixing
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
		ci_arr=($(${SUDO} nvidia-smi mig -lci | awk '{print $2}' | grep -P '^\d+$'))
		gi_arr=($(${SUDO} nvidia-smi mig -lci | awk '{print $3}' | grep -P '^\d+$'))
		i=0
		while [[ $i -lt ${#gi_arr[*]} ]];
		do
			ci=${ci_arr[$i]}
			gi=${gi_arr[$i]}
			${SUDO} nvidia-smi mig -dci -ci ${ci} -gi ${gi}
			i=$(( $i + 1))
		done

		# Cleanup GPU Instances
		${SUDO} nvidia-smi mig -i ${device_id} -dgi
		exit_code=$?
		if [[ ${exit_code} -eq 0 || ${exit_code} -eq 6 ]]; then
			break
		fi
		print "${SUDO} nvidia-smi mig -i ${device_id} -dgi failed. Trying in 1s"
		sleep 1
	done
}

function cleanup()
{
    mode=$1
    device_id=$2
    disable_mps_if_needed ${mode} ${device_id}
    cleanup_mig_if_needed ${mode} ${device_id}
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
    local WS=$(git rev-parse --show-toplevel)
    local DOCKER_WS=/root/$(basename ${WS})

    # Setup container
    ${DOCKER} rm -f ${ORION_CTR} >/dev/null 2>&1 || :
    ${DOCKER} run -v ${WS}:${DOCKER_WS} -it -d \
        -v /tmp:/tmp \
        -w ${DOCKER_WS} \
        --name ${ORION_CTR} \
        --ipc=host --pid=host \
        --gpus "device=${device_id}" \
        ${ORION_IMG} bash > /dev/null

    # Install necessary package
    NSIGHT_COMPUTE_TAR=nsight-compute.tar
    if [[ ! -f ${NSIGHT_COMPUTE_TAR} ]]; then
        # pip install gdown
        cmd="gdown --id 1_HY1FOIS6KP7dLTKRZ30Wliqc9P_N7hu"
        eval ${cmd}
        exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            echo "gdown package not found!"
            echo "Install using: 'pip install gdown'"
            echo "Or try running command: '${cmd}'"
            return
        fi
    fi
    ${DOCKER} cp ${NSIGHT_COMPUTE_TAR} ${ORION_CTR}:/usr/local/ > /dev/null 2>&1
    ${DOCKER} exec -it ${ORION_CTR} bash -c "tar -xf /usr/local/nsight-compute.tar -C /usr/local/ > /dev/null 2>&1"
    ${DOCKER} exec -it ${ORION_CTR} bash -c "wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1"
    ${DOCKER} exec -it ${ORION_CTR} bash -c "dpkg -i nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb > /dev/null 2>&1"
    ${DOCKER} exec -it ${ORION_CTR} bash -c "pip install transformers > /dev/null 2>&1"
}

function setup_tie_breaker_container {
    local device_id=$1
    local WS=$(git rev-parse --show-toplevel)
    local DOCKER_WS=/home/$USER/$(basename ${WS})

    # Setup container
    ${DOCKER} rm -f ${TIE_BREAKER_CTR} >/dev/null 2>&1 || :
    ${DOCKER} run -v ${WS}:${DOCKER_WS} -it -d \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /tmp:/tmp \
        -w ${DOCKER_WS} \
        -u $(id -u $USER):$(id -g $USER) \
        --name ${TIE_BREAKER_CTR} \
        --ipc=host --pid=host \
        --gpus "device=${device_id}" \
        ${TIE_BREAKER_IMG} bash > /dev/null
    sleep 2
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


mps_mig_percentages=("" "100" "57,43" "42,29,29" "29,29,28,14" "29,29,14,14,14" "29,15,14,14,14,14" "15,15,14,14,14,14,14")
mps_equi_percentages=("" "100" "50,50" "34,33,33" "25,25,25,25" "20,20,20,20,20" "17,17,17,17,16,16" "15,15,14,14,14,14,14")
mps_chunks_percentages=("" "14" "14,14" "14,14,14" "14,14,14,14" "14,14,14,14,14" "14,14,14,14,14,14" "14,14,14,14,14,14,14")
attempts=10
# Command to profile
# nsys profile --stats=true --force-overwrite true --wait=all -o trial
