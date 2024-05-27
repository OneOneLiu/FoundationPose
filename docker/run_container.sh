docker rm -f foundationpose
DIR=$(pwd)/../
xhost +
XAUTH=/tmp/.docker.xauth
docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --gpus all \
    --network=host \
    --name foundationpose  \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp  \
    --ipc=host \
    -e GIT_INDEX_FILE foundationpose:latest bash -c "cd $DIR && bash"