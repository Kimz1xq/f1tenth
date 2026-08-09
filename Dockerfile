# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:humble

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       python3-pip \
                       libeigen3-dev \
                       tmux \
                       ros-humble-rviz2
RUN apt-get -y dist-upgrade

# Python dependencies for f1tenth_gym
# Use apt for large scientific packages to avoid PyPI wheel hash/download issues.
RUN apt-get update --fix-missing && \
    apt-get install -y \
        python3-scipy \
        python3-numba \
        python3-pil \
        python3-opengl \
        python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

# Small Python packages from pip
RUN python3 -m pip install --no-cache-dir \
    gym==0.19.0 \
    pyglet==1.4.11 \
    cloudpickle==1.6.0 \
    future \
    transforms3d \
    setuptools==69.5.1 \
    cvxpy==1.3.2 \
    osqp==0.6.3

RUN git clone https://github.com/f1tenth/f1tenth_gym
RUN cd f1tenth_gym && \
    python3 -m pip install --no-cache-dir -e . --no-deps

RUN mkdir -p sim_ws/src/f1tenth_gym_ros
COPY . /sim_ws/src/f1tenth_gym_ros
RUN source /opt/ros/humble/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro humble -y && \
    colcon build

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]

# 추가 패키지
RUN apt-get update && apt-get install -y \
    ros-humble-slam-toolbox \
    ros-humble-nav2-amcl \
    ros-humble-nav2-rviz-plugins \
    ros-humble-nav2-map-server \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-ackermann-msgs \
    ros-humble-teleop-twist-keyboard \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-cartographer \
    ros-humble-cartographer-ros && \
    python3 -m pip install --no-cache-dir casadi==3.7.2 --no-deps
