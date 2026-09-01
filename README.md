## Enter normal
srun --partition=L40 --gpus=1 --pty --container-image='nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04' --container-workdir="${PWD}" --container-name='slm' bash

## Enter as root
srun --container-name=slm --container-remap-root --pty bash

## List containeres
srun --pty enroot list

## Delete containers
srun --pty enroot remove pyxis_slm


