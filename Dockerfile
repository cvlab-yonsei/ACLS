ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
MAINTAINER <hyekang.park@yonsei.ac.kr>

# Update and install
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y \
      git \
      vim \
      zsh \
      byobu \
      htop \
      curl \
      wget \
      locales \
      zip

# Install language pack
RUN apt-get install -y language-pack-en
RUN locale-gen en_US.utf8
RUN update-locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# image pack
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libpng-dev
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libjpeg8-dev

#################################################################################################################################################################
RUN pip install --upgrade pip setuptools wheel
RUN pip install jupyter numpy scipy ipython pandas opencv-python matplotlib scikit-image scikit-learn tensorboard numba PyYAML plotly h5py easydict
RUN pip install --upgrade ipykernel

RUN apt-get install -y libboost-all-dev
RUN pip install cmake

#################################################################################################################################################################

# ZSH Theme Setting
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)" -- \
    -t agnoster \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/zsh-users/zsh-history-substring-search

# Set ZSH for default in byobu
RUN mkdir -p /root/.byobu/
RUN echo "set -g default-shell /usr/bin/zsh" >> /root/.byobu/.tmux.conf
RUN echo "set -g default-command /usr/bin/zsh" >> /root/.byobu/.tmux.conf
RUN echo "set -g mouse on" >> /root/.byobu/.tmux.conf
RUN echo "set -g mouse-select-pane on" >> /root/.byobu/.tmux.conf
RUN echo "set -g mouse-select-window on" >> /root/.byobu/.tmux.conf
RUN echo "set -g mouse-resize-pane on" >> /root/.byobu/.tmux.conf
RUN echo "set -g mouse-utf8 on" >> /root/.byobu/.tmux.conf

#CleanUp
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

CMD ["zsh"]