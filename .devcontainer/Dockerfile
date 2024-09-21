# Base image
FROM docker.io/redhat/ubi9:latest

# Install Python, Git, and pip
RUN yum install -y python3 git python3-pip && \
    yum clean all

# Set the working directory
WORKDIR /workspace

# Copy the necessary files from the parent directory
COPY ../setup.py .
COPY ../setup.cfg .
COPY ../pyproject.toml .

# Upgrade pip and install the package with development dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .[dev]

# Set the default command to run when the container starts
CMD ["python3"]