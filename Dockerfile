FROM python:3.11-slim
WORKDIR /usr/src/app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y git ninja-build ccache libopenblas-dev libopencv-dev cmake && \
    apt-get install -y gcc mono-mcs g++ && \
    apt-get install -y default-jdk && \
    apt-get install -y libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install & install requirements
COPY requirements-dev0.txt ./requirements-dev0.txt
COPY requirements-dev1.txt ./requirements-dev1.txt
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements-dev0.txt
RUN pip install -r requirements-dev1.txt
RUN pip install -r requirements.txt

# Copy library source
COPY modelstore ./modelstore
COPY tests ./tests

# Run tests
ENTRYPOINT ["python3", "-m", "pytest", "--exitfirst", "./tests"]
