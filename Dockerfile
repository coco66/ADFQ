FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install sudo git wget python-dev python3-dev libopenmpi-dev python-pip python3-pip zlib1g-dev cmake python-opencv tmux libav-tools vim

RUN pip3 install --upgrade pip

RUN pip3 install pyyaml scipy numpy tabulate tensorflow-gpu matplotlib
RUN pip3 install gym[mujoco,atari,classic_control,robotics] tqdm joblib zmq dill progressbar2 mpi4py cloudpickle click opencv-python

WORKDIR /tmp/

RUN export uid=1008 gid=1008 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer

ENV HOME /home/developer/Desktop/docker-code/
WORKDIR /home/developer/Desktop/docker-code/

