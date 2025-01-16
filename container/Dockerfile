# Use the NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    software-properties-common \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libffi-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libexpat1-dev \
    libxerces-c-dev \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libspatialite-dev \
    libgl1 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libexpat1-dev \
    libxerces-c-dev \
    zlib1g-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libspatialite-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and build GDAL 3.5.3 from source
RUN wget https://download.osgeo.org/gdal/3.5.3/gdal-3.5.3.tar.gz -O /tmp/gdal-3.5.3.tar.gz && \
    tar -xzf /tmp/gdal-3.5.3.tar.gz -C /tmp && \
    cd /tmp/gdal-3.5.3 && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/gdal-3.5.3 /tmp/gdal-3.5.3.tar.gz

# Verify GDAL installation
RUN gdalinfo --version

# Set environment variables to help rasterio find GDAL
ENV CPLUS_INCLUDE_PATH=/usr/local/include
ENV C_INCLUDE_PATH=/usr/local/include
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements.txt into the container
COPY requirements.txt /tmp/

# Install Python dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install MapReader from GitHub without dependencies to prevent altering existing packages
RUN pip3 install git+https://github.com/Living-with-machines/MapReader.git --no-deps

# Set the working directory
WORKDIR /workspace

# Define the default command to run when starting the container
CMD ["/bin/bash"]
