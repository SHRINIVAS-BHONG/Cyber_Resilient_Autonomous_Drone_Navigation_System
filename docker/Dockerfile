# Dockerfile for Cyber-Resilient Autonomous Drone Navigation System
# Target OS: Ubuntu 22.04 LTS (Jammy) with ROS 2 Humble Desktop

FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble
ENV WORKSPACE=/workspace/cyber_drone

# 1. Install system utilities and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    net-tools \
    iproute2 \
    iputils-ping \
    nano \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 2. Build Micro-XRCE-DDS-Agent (PX4 SITL bridge to ROS 2)
WORKDIR /tmp
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    && cd Micro-XRCE-DDS-Agent \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig /usr/local/lib/ \
    && rm -rf /tmp/Micro-XRCE-DDS-Agent

# 3. Install Python scientific, robotics, and web telemetry dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    joblib \
    filterpy \
    pyyaml \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    streamlit \
    plotly \
    pytest \
    pytest-cov \
    networkx \
    rosbags

# 4. Create non-root developer user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 5. Set up workspace directory
WORKDIR $WORKSPACE
RUN chown -R $USERNAME:$USERNAME $WORKSPACE

# 6. Configure environment and entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $USERNAME

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
