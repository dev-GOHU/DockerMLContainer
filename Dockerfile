ARG OS_NAME=ubuntu
ARG OS_MAJOR=22
ARG OS_MINOR=04

ARG CUDA_MAJOR=11
ARG CUDA_MINOR=8
ARG CUDA_PATCHLEVEL=0

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.${CUDA_PATCHLEVEL}-devel-${OS_NAME}${OS_MAJOR}.${OS_MINOR}

####################
### set cuda env ###
####################
ENV CUDA_DIR /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
ENV PATH $CUDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH $CUDA_DIR/lib64:$LD_LIBRARY_PATH

#############################
# Set up time zone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
############################

ARG DEBIAN_FRONTEND=noninteractive

# Install requirements
RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    wget \
    curl \
    software-properties-common \
    git \
    apt-transport-https \
    gnupg \
    lsb-core \
    nano \
    sudo \
    zlib1g-dev \
    ssh \
    ca-certificates \
    cmake

#########################

# Install bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
    mv bazel-archive-keyring.gpg /usr/share/keyrings && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt-get update -y && apt-get install -y bazel

# Install llvm
ARG LLVM_MAJOR=16
RUN wget -q https://apt.llvm.org/llvm.sh -O ~/llvm.sh&& \
    chmod +x ~/llvm.sh && \
    ~/llvm.sh ${LLVM_MAJOR}
RUN rm ~/llvm.sh
ARG LLVM_MAJOR
ENV LLVM_DIR /usr/lib/llvm-${LLVM_MAJOR}
ENV PATH $LLVM_DIR/bin:$PATH
ENV LD_LIBRARY_PATH $LLVM_DIR/lib:$LD_LIBRARY_PATH

# Install OpenCV
RUN apt-get install -y unzip \
    # For access image file
    libjpeg-dev libtiff5-dev libpng-dev \
    # For access video file
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
    # For supporting instantly video capture
    libv4l-dev v4l-utils \
    # For video streaming
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    # For supporting OpenGL
    mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev \
    # For Optimizing OpenCV
    libatlas-base-dev gfortran libeigen3-dev \
    # Install OpenCV
    libopencv-dev


#####################
### Install cudnn ###
#####################
ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG CUDNN_MAJOR=8
ARG CUDNN_MINOR=6
ARG CUDNN_PATCHLEVEL=0
RUN apt-get install -y \
    libcudnn${CUDNN_MAJOR}=${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libcudnn${CUDNN_MAJOR}-dev=${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR}

#### Installed in /usr/include/ 
#### You can check cudnn version by "cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2"

########################
### Install TensorRT ###
########################
ARG PYTHON_MAJOR=3
ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG TENSORRT_MAJOR=8
ARG TENSORRT_MINOR=6
ARG TENSORRT_PATCHLEVEL=1
RUN apt-get install -y \
    libnvinfer-headers-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvinfer-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer-lean${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvinfer-lean-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer-dispatch${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvinfer-dispatch-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer-headers-plugin-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer-plugin${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvinfer-plugin-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvinfer-vc-plugin${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvinfer-vc-plugin-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvparsers${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvparsers-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    libnvonnxparsers${TENSORRT_MAJOR}=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libnvonnxparsers-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    ###################
    python${PYTHON_MAJOR}-libnvinfer=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    python${PYTHON_MAJOR}-libnvinfer-lean=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    python${PYTHON_MAJOR}-libnvinfer-dispatch=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    python${PYTHON_MAJOR}-libnvinfer-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR}

# install tensorrt
RUN apt-get install -y tensorrt-dev=${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCHLEVEL}.*-1+cuda${CUDA_MAJOR}.${CUDA_MINOR}

#########################
### Install miniconda ###
#########################
ENV CONDA_DIR $HOME/.conda

RUN mkdir -p $HOME/.conda && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/Miniconda.sh && \
    sh ~/Miniconda.sh -b -u -p ${CONDA_DIR} && \
    rm ~/Miniconda.sh

ENV PATH ${CONDA_DIR}/bin:$PATH
ENV LD_LIBRARY_PATH ${CONDA_DIR}/lib64:$LD_LIBRARY_PATH
# Set conda channel default to conda-forge
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

##########################
### Set ML environment ###
##########################

# If you want to use TensorRT with the UFF converter to convert models from TensorFlow
RUN conda run -n base pip install protobuf && \
    apt-get install -y uff-converter-tf

# If you want to run samples that require onnx-graphsurgeon or use the Python module for your own project
RUN conda run -n base pip install numpy onnx && \
    apt-get install onnx-graphsurgeon

# Add virtual environment
RUN conda run -n base pip install numpy scipy sympy pandas matplotlib seaborn scikit-learn opencv-python

# Install tensorflow
ARG TENSORFLOW_MAJOR=2
ARG TENSORFLOW_MINOR=13
RUN conda run -n base pip install tensorflow==${TENSORFLOW_MAJOR}.${TENSORFLOW_MINOR}.* && \
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Install tensorboard
RUN conda run -n base pip install tensorboard==${TENSORFLOW_MAJOR}.${TENSORFLOW_MINOR}.*
# Install pytorch
ARG CUDA_MAJOR
ARG CUDA_MINOR
RUN conda run -n base pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

RUN conda run -n base conda install -y -c conda-forge jupyterlab

ENV CONTAINER_SHELL bash
RUN conda init ${CONTAINER_SHELL}

WORKDIR $HOME