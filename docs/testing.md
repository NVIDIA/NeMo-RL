# Testing NeMo-Reinforcer

## Unit Tests

```sh
uv pip install -e '.[test]'
uv run bash tests/run_unit.sh 
```

### Run Unit Tests Hermetic

If your local environment does not have all the necessary dependencies (e.g., `gcc`, `nvcc`)
or there is concern that something in your environment may be misconfigured, you can also run
the tests in docker with this script:

```sh
CONTAINER=... bash tests/run_unit_in_docker.sh
```

The `CONTAINER` can be built by following the instructions [here](docker.md).

## Functional tests

TBD
