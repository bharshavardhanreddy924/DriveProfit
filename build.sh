#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    python3-dev \
    libtiff5 \
    libjpeg8-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk

# Set locale to prevent build errors
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader vader_lexicon