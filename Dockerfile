FROM ubuntu
WORKDIR /usr/src/app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y git ninja-build ccache libopenblas-dev libopencv-dev cmake && \
    apt-get install -y gcc mono-mcs g++ && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools

# Install & install requirements
COPY requirements-dev0.txt ./requirements-dev0.txt
COPY requirements-dev1.txt ./requirements-dev1.txt
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements-dev0.txt
RUN pip3 install -r requirements-dev1.txt
RUN pip3 install -r requirements.txt

# Copy library source
COPY modelstore ./modelstore
COPY tests ./tests

# Run tests
ENTRYPOINT ["python3", "-m", "pytest", "--exitfirst", "./tests"]
