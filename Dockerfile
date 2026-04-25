# Use Ubuntu 22.04 as base
FROM ubuntu:22.04

# Set environment variables to ensure non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive


# Update and install required packages
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9 python3.9-venv python3.9-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN python3 -m pip install aalpy==1.3.2
RUN python3 -m pip install automata-lib==8.4.0
RUN python3 -m pip install z3-solver==4.13.0.0
RUN python3 -m pip install numpy==2.0.0
RUN python3 -m pip install pandas==2.2.2
RUN python3 -m pip install stopit==1.1.2

RUN apt-get install -y openjdk-11-jdk
RUN apt-get install -y git


# download repo
RUN git clone https://github.com/bThink-BGU/Papers-2025-MODELS-Automata-Bug-Description.git

# compile java files
RUN javac -d Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019 Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/DelayWrapper.java

# compile "m24", "m45", "m54", "m55", "m76", "m95", "m135", "m158", "m159", "m164", "m172", "m181", "m183", "m185", "m201", "m22", "m27", "m41", "m106", "m131", "m132", "m167", "m173", "m182", "m189", "m196", "m199"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m24"
RUN javac m24_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m45"
RUN javac m45_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m54"
RUN javac m54_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m55"
RUN javac m55_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m76"
RUN javac m76_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m95"
RUN javac m95_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m135"
RUN javac m135_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m158"
RUN javac m158_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m159"
RUN javac m159_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m164"
RUN javac m164_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m172"
RUN javac m172_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m181"
RUN javac m181_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m183"
RUN javac m183_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m185"
RUN javac m185_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/arithmetic/m201"
RUN javac m201_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m22"
RUN javac m22_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m27"
RUN javac m27_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m41"
RUN javac m41_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m106"
RUN javac m106_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m131"
RUN javac m131_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m132"
RUN javac m132_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m167"
RUN javac m167_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m173"
RUN javac m173_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m182"
RUN javac m182_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m189"
RUN javac m189_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m196"
RUN javac m196_Reach.java
WORKDIR "/"

WORKDIR "Papers-2025-MODELS-Automata-Bug-Description/rers2019/IndReachabilityRers2019/data-structures/m199"
RUN javac m199_Reach.java
WORKDIR "/"

# Cleanup to reduce image size
RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && rm get-pip.py

# Define a default command, for this case, we're just using a bash shell
WORKDIR "/Papers-2025-MODELS-Automata-Bug-Description"
#ENTRYPOINT git pull &> /dev/null && /bin/bash

RUN pip install pysmt
RUN git clone https://github.com/JLaumen/l-sharp-square-algorithm.git -b "rers"
RUN mv -f ./l-sharp-square-algorithm/* ./

CMD ["/bin/bash"]
