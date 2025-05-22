# Build Docker Images

This guide provides two methods for building Docker images: the base image, ideal for specifying Python dependencies at runtime, and the hermetic image, which includes default dependencies for offline use.

## Base Image

If you only need the base image with ray + uv, you can build it like so:

```sh
cd docker/
docker buildx build --target base -t nemo_rl -f Dockerfile ..
```

This is **our recommendation** as it is a small image and allows you to specify your Python dependencies at runtime.

## Hermetic Image

The Docker image build without a target stage will include all of the default dependencies to get started.

```sh
cd docker/
docker buildx build -t nemo_rl -f Dockerfile ..
```

This image sets up the Python environment for you, so you do not have to use `uv` if you don't need
any other packages.

This image is useful in situations where you may not have network connectivity to re-download packages.
