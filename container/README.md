# Run MapReader on Isambard-AI

This document helps you to run MapReader on [Isambard-AI](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-1).

## Set-up

1. Get familar with [Podman-HPC at Isambard](https://docs.isambard.ac.uk/user-documentation/guides/containers/podman-hpc/).

2. Clone MapReader and navigate to the container directory.

```bash
git clone https://github.com/maps-as-data/MapReader.git
cd MapReader/container
```
## Build and run the image

### Build the image

After navigating the `container` directory in the MapReader repo, run the following command to build the image:

```bash
podman-hpc build -t mapreader .
```

If you run into any issues with this, see the ```Pull the image``` section at the bottom of this document.

### Migrate the image

Migration is the process of moving the image to the shared filesystem.
This is needed to run the MapReader image on the compute nodes.

To migrate the image, run the following command:

```bash
podman-hpc migrate mapreader:latest
```

### Run as a batch job

To run the MapReader image as part of a batch job, you will need a slurm script.
Ours looks like this:


To run the script, run the following command:

```bash
sbatch pytorch_run_podmanhpc.sh
```

### Run in an interactive shell

If instead you'd like to run in an interactive shell, you can use:

```bash
podman-hpc run -it --gpu --rm --name mapreader \
  localhost/mapreader:latest /bin/bash
```

If you'd like to do this using GPU, you will need to launch an interactive job first using srun:

```bash
srun --gres=gpu:1 -A <project> --time 1:00:00 --pty /bin/bash
```

Then you can run the podman-hpc command above, the `--gpu` flag will ensure that your requested GPUs are available to the container.

### Run the dependency test

To check everything is working as expected, run the dependency test using the following command:

```bash
python test.py
```

## Pull the image

If you have issues building the image, please to pull it from the [docker hub](https://hub.docker.com/repository/docker/oxfordfun/mapreader/tags)

```bash
podman-hpc pull oxfordfun/mapreader:3.5.3-f
```

Then go on to migrate and run the image as described above.
