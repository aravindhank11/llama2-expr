### Setup Environment
* Setup python virtual environment:
    * `[sudo] apt install python3-venv`
    * `python3 -m venv tie-breaker-venv`
    * `source tie-breaker-venv/bin/activate`
    * `pip install -r packaging/requirements.txt`
* Follow the below steps if Benchmarking against Orion (Optional):
    * Perform the following to enable benchmarking for Orion on the host:
    ```
    sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'
    sudo update-initramfs -u -k all
    sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
    ```
    * Ref: [nvidia-developer-forum](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)


### How to run experiments
```
export USE_SUDO=1  # If you want sudo to enable MPS, MIG, and use docker

# Use python virtual environment
export VENV=tie-breaker-venv

# Use it to run closed loop experiment
# Once results for closed loop is obtained, use it to run for various loads of other distribution types
./run.sh --help

# Runs a job mix for a particular configuration (Used by run.sh -- but is standalone as well)
./run_job_mix.sh --help
```


### To generate new configuration file for [Orion](Orion)
```
export USE_SUDO=1
source helper.sh
setup_orion_container <GPU_DEVICE_ID>
[sudo] docker exec -it orion bash
./prepare_orion_input_config.sh <v100 | a100 | h100> <device_id> <model> <batchsize> <vision | bert | transformer>
```
