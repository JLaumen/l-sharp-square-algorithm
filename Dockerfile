# Ubuntu 26.04 LTS
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.14.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-venv \
        python3.14-dev \
        openjdk-11-jdk \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment.
RUN python3.14 -m venv /venv

ENV PATH=/venv/bin:$PATH

# Install Python dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        aalpy==1.3.2 \
        automata-lib==8.4.0 \
        python-sat \
        numpy==2.5.3 \
        pandas==3.0.6 \
        stopit==1.1.2 \

# Download repo.
RUN git clone \
    https://github.com/bThink-BGU/Papers-2025-MODELS-Automata-Bug-Description.git

# Compile Java files.
RUN javac \
    -d Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019 \
    Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/DelayWrapper.java

# Compile arithmetic benchmarks.

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m24
RUN javac m24_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m45
RUN javac m45_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m54
RUN javac m54_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m55
RUN javac m55_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m76
RUN javac m76_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m95
RUN javac m95_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m135
RUN javac m135_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m158
RUN javac m158_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m159
RUN javac m159_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m164
RUN javac m164_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m172
RUN javac m172_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m181
RUN javac m181_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m183
RUN javac m183_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m185
RUN javac m185_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m201
RUN javac m201_Reach.java

# Compile data-structure benchmarks.

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m22
RUN javac m22_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m27
RUN javac m27_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m41
RUN javac m41_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m106
RUN javac m106_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m131
RUN javac m131_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m132
RUN javac m132_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m167
RUN javac m167_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m173
RUN javac m173_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m182
RUN javac m182_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m189
RUN javac m189_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m196
RUN javac m196_Reach.java

WORKDIR /Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m199
RUN javac m199_Reach.java

# Clone the L#-square repository.
WORKDIR /
RUN git clone \
    https://github.com/JLaumen/l-sharp-square-algorithm.git \
    -b rers

# Merge L#-square into the RERS repository.
RUN mv -f \
    ./l-sharp-square-algorithm/* \
    ./Papers-2025-MODELS-Automata-Bug-Description/

# Final working directory.
WORKDIR /Papers-2025-MODELS-Automata-Bug-Description

CMD ["/bin/bash"]