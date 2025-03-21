# Testing Reinforcer

## Unit Tests

:::{important}
Unit tests require 2 GPUs to test the full suite.
:::

```sh
# Install the project and the test dependencies
uv pip install -e '.[test]'

# Run the unit tests using local GPUs
uv run bash tests/run_unit.sh 
```

:::{note}
Tests can also be run on SLURM with `ray.sub`, but note that some tests will be skipped
due to no GPUs being located on the head node. To run the full suite of tests, please
launch on a regular GPU allocation.
:::

### Running Unit Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in docker with this script:

```sh
CONTAINER=... bash tests/run_unit_in_docker.sh
```

The required `CONTAINER` can be built by following the instructions in the [docker documentation](docker.md).

## Functional tests

:::{important}
Functional tests may require multiple GPUs to run. See each script to understand the requirements.
:::

Functional tests are located under `tests/functional/`.

```sh
# Install the project and the test dependencies
uv pip install -e '.[test]'
# Run the functional test for sft
uv run bash tests/functional/sft.sh
```

At the end of each functional test, the metric checks will be printed as well as
whether they pass or fail. Here is an example:

```text
                              Metric Checks
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Status ┃ Check                          ┃ Value             ┃ Message ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ PASS   │ data["train/loss"]["9"] < 1500 │ 817.4517822265625 │         │
└────────┴────────────────────────────────┴───────────────────┴─────────┘
```

### Running Functional Tests in a Hermetic Environment

For environments lacking necessary dependencies (e.g., `gcc`, `nvcc`)
or where environmental configuration may be problematic, tests can be run
in docker with this script:

```sh
CONTAINER=... bash run_functional_in_docker.sh functional/sft.sh
```
