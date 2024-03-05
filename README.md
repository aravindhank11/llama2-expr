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
WS=$(git rev-parse --show-toplevel)
DOCKER_WS=/root/$(basename ${WS})
export USE_SUDO=1 # If you need sudo privileges to run docker commands (Optional)
source helper.sh
setup_tie_breaker_container
${DOCKER} exec -it ${TIE_BREAKER_CTR} bash -c "cd ${DOCKER_WS} && bash"
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
