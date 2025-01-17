# Run MapReader on Isambard-AI

This document helps you to run MapReader on [Isambard-AI](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-1).

## Prep

1. Get familar with [Podman-HPC at Isambard](https://docs.isambard.ac.uk/user-documentation/guides/containers/podman-hpc/).

2. Get this branch

```bash
git clone https://github.com/owhere/MapReader.git
git checkout isambard
cd container
```
## Build and Run the image

### Build the image

```bash
podman-hpc build -t mapreader .
```

### Run in an interactive shell
```bash
podman-hpc run -it --gpu --rm --name mapreader \
  localhost/mapreader:latest /bin/bash
```

### Run the dependency test
```bash
python test.py
```

