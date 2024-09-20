# .devcontainer/Dockerfile
FROM docker.io/redhat/ubi9:latest

# Install Python and Git
RUN yum install -y python3 git && \
    yum clean all

# Set the working directory
WORKDIR /workspace

# Command to run your application (adjust as necessary)
CMD ["python3"]

