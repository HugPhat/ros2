ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.3
FROM ${BASE_IMAGE}

ARG ROS_PKG=ros_base
ENV ROS_DISTRO=eloquent
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# change the locale from POSIX to UTF-8
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# add the ROS deb repo to the apt sources list
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    curl \
    wget \ 
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc && apt-key add ros.asc
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# install ROS packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ros-eloquent-`echo "${ROS_PKG}" | tr '_' '-'` \
    ros-eloquent-launch-xml \
    ros-eloquent-launch-yaml \
    ros-eloquent-launch-testing \
    ros-eloquent-launch-testing-ament-cmake \
    ros-eloquent-camera-calibration-parsers \
    ros-eloquent-camera-info-manager \
    ros-eloquent-cv-bridge \
    ros-eloquent-v4l2-camera \
    ros-eloquent-vision-msgs \
    ros-eloquent-vision-opencv \
    ros-eloquent-image-transport \
    ros-eloquent-image-tools \
    ros-eloquent-image-geometry \
    ros-eloquent-gazebo-ros \
    ros-eloquent-gazebo-msgs \
    ros-eloquent-gazebo-ros-pkgs \
    ros-eloquent-gazebo-plugins \
    libpython3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    libgazebo9-dev \
    gazebo9 \
    gazebo9-common \
    gazebo9-plugin-base \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# init/update rosdep
RUN apt-get update && \
    cd ${ROS_ROOT} && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*

# compile yaml-cpp-0.6, which some ROS packages may use (but is not in the 18.04 apt repo)
RUN git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
    cd yaml-cpp-0.6 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    cp libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/ && \
    ln -s /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6

# setup entrypoint
COPY ./packages/ros_entrypoint.sh /ros_entrypoint.sh
RUN echo 'source ${ROS_ROOT}/setup.bash' >> /root/.bashrc 
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /

#
# install OpenCV (with GStreamer support)
#
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# PyTorch Installations
# ----------------------------
#
# install prerequisites (many of these are for numpy)
#
ENV PATH="/usr/local/cuda-10.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"


RUN apt-get update && \
    ldconfig && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libopenmpi2 \
    openmpi-bin \
    openmpi-common \
    gfortran \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install numpy --verbose
# 
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"
RUN pip3 install --upgrade setuptools
RUN pip3 install pycuda --verbose

#---------------------
# cv_bridge dependency
#---------------------

RUN apt-get update
RUN apt-get install -y cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev
RUN apt-get install -y vim

RUN echo 'source ${ROS_ROOT}/setup.bash'
COPY ./trt_yolov5 /workspace/src/trt_yolov5
WORKDIR /workspace/
COPY *.sh /workspace/
#RUN rosdep install -i --from-path src --rosdistro eloquent -y

CMD ["bash"]

