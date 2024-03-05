#!/bin/bash -xe

# Enable profiling
sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia.conf'
sudo update-initramfs -u -k all
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
