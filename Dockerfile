FROM ubuntu
WORKDIR /usr/src/app

# Install gcc & python
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    apt-get -y install g++ && \
    apt-get -y install python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip & setuptools
RUN pip3 install --upgrade pip setuptools

# Install & install requirements
ADD requirements*.txt ./
RUN pip3 install -r requirements-dev0.txt
RUN pip3 install -r requirements-dev1.txt
RUN pip3 install -r requirements.txt

# Copy library source
COPY modelstore ./
COPY tests ./

# Run tests
RUN ["python3.8", "-m", "pytest", "tests"]
