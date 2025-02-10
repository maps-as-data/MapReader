
#! /bin/bash

# podman-hpc build -t mapreader .

podman-hpc run -it --gpu --rm --name mapreader \
  localhost/mapreader:latest /bin/bash
