FROM ros:jazzy-ros-base-noble

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    ros-jazzy-rviz2 \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ament-lint-auto \
    ros-jazzy-ament-lint-common \
    liburdfdom-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /raptor_ws
CMD ["bash"]
