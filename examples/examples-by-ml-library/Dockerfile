FROM python:3.11-slim
WORKDIR /usr/src/app

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake && \
    apt-get install -y gcc mono-mcs g++ && \
    apt-get install -y default-jdk && \
    apt-get install -y libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# Install modelstore library from test.pypi.org
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                modelstore

# Install example dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy example source
COPY . ./examples
WORKDIR /usr/src/app/examples

ENTRYPOINT ["python", "main.py"]
