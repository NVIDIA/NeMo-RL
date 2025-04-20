# Building Docker Images

## Base Image

If you only need the base image with ray + uv, you can build it like so:

```sh
cd docker/
docker buildx build --target base -t reinforcer -f Dockerfile ..
```

This is **our recommendation** as it is a small image and allows you to specify your python dependencies at runtime.

## Hermetic Image

The docker image build without a target stage will include all of the default dependencies to get started.

```sh
cd docker/
docker buildx build -t reinforcer -f Dockerfile ..
```

This image sets up the python environment for you, so you do not have to use `uv` if you don't need
any other packages.

This image is useful in situations where you may not have network connectivity to re-download packages.

# SquashFS Container

To run reinforcer on a Slurm cluster, you need to build a SquashFS Container from the base docker image.

Follow [these installation commands](https://github.com/NVIDIA/enroot/blob/master/doc/installation.md#standard-flavor) to install enroot. Then build the container with the following command.

```sh
enroot import -o reinforcer.sqsh dockerd://reinforcer
```