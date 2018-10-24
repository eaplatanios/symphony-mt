# Running on Google Compute Engine

## Installation

```bash
#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

### Install CUDA
echo "Checking for CUDA and installing."

# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-2; then
  # The 16.04 installer works with 16.10.
  # Adds NVIDIA package repository.
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
  wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
  sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
  sudo apt-get update
  # Includes optional NCCL 2.x.
  sudo apt-get install cuda9.2 cuda-cublas-9-2 cuda-cufft-9-2 cuda-curand-9-2 \
    cuda-cusolver-9-2 cuda-cusparse-9-2 libcudnn7=7.1.4.18-1+cuda9.2 \
     libnccl2=2.2.13-1+cuda9.2 cuda-command-line-tools-9-2
  # Optionally install TensorRT runtime, must be done after above cuda install.
  sudo apt-get update
  sudo apt-get install libnvinfer4=4.1.2-1+cuda9.2
fi

# Enable persistence mode
nvidia-smi -pm 1

### Install SBT
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt

### Install Utilities
sudo apt-get install htop
```
