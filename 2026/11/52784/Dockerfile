FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
COPY cuda-keyring.deb /tmp/
RUN dpkg -i /tmp/cuda-keyring.deb && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnccl2=2.32.3-1+cuda13.4 cuda-cudart-13-4 libibverbs1 ibverbs-providers perftest iproute2 && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb
COPY nccl-check /usr/local/bin/nccl-check
ENV NVIDIA_VISIBLE_DEVICES=void NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["sleep", "infinity"]
