FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/JLaumen/l-sharp-square-algorithm.git --single-branch

WORKDIR /app/l-sharp-square-algorithm

RUN pip3 install --no-cache-dir -r requirements.txt

# Default to interactive shell
CMD ["/bin/bash"]
