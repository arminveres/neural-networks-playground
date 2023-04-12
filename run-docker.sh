#!/bin/bash

set -ev

docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G \
    -v ./:/neural-networks\
    rocm/pytorch:latest
