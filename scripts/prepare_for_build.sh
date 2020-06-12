#!/usr/bin/env bash

#sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
#sudo apt-get -q update
#sudo apt-get -y install gcc-4.8
apt-get -y install libfreetype6-dev
apt-get -y install pkg-config
apt-get -y install libpng12-dev
apt-get -y install pkg-config

pip install Cython
