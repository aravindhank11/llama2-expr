### Enabling profiling
* The following steps need to be run on the host to enable profiling
* This is a 1 time setup activity
```
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'
sudo update-initramfs -u -k all
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```
Ref: [nvidia-developer-forum](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
* Follow OS specific instructions to install 'nvidia-container-toolkit'


### How to run experiments
```
# Given a model mix runs experiment for 100% load using closed loop
# Using the obtained results runs for various load mixes
./run.sh --help

# Runs a job mix for a particular configuration (Used by run.sh -- but is standalone as well)
./run_expr.sh --help
```


### To build your own docker image (Optional)
```
# NOTE: Make sure to change the variable `TIE_BREAKER_IMG`
export USE_SUDO=1 # If you need sudo privileges to run docker commands (Optional)
source helper.sh
cd packaging
./build.sh
```


### To generate new configuration file for orion
```
export USE_SUDO=1
source helper.sh
setup_orion_container <GPU_DEVICE_ID>
[sudo] docker exec -it orion bash
./prepare_orion_input_config.sh <v100 | a100 | h100> <device_id> <model> <batchsize> <vision | bert | transformer>
```
