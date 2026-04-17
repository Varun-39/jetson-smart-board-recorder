# Must use the official L4T ML image for Python 3.6 / JetPack 4.6.x
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

# Install system dependencies via apt-get for GStreamer and sqlite3
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement list and evaluate
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files (including main.py, db_setup.py, etc)
COPY . /app/

# Expose Web port
EXPOSE 5000

# Pre-initialize Database then boot main python loop
CMD python3 db_setup.py && python3 main.py
