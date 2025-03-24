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

### Tracking metrics in unit tests

Unit tests may also log metrics to a fixture. The fixture is called `tracker` and has the following API:
```python
# Track an arbitrary metric (must be json serializable)
tracker.track(metric_name, metric_value)
# Log the maximum memory across the entire cluster. Okay for tests since they are run serially.
tracker.log_max_mem(metric_name)
# Returns the maximum memory. Useful if you are measuring changes in memory.
tracker.get_max_mem()
```

Including the `tracker` fixture also tracks the elapsed time for the test implicitly.

Here is an example test:
```python
def test_exponentiate(tracker):
    starting_mem = tracker.get_max_mem()
    base = 2
    exponent = 4
    result = base ** exponent
    tracker.track("result", result)
    tracker.log_max_mem("memory_after_exponentiating")
    change_in_mem = tracker.get_max_mem() - starting_mem
    tracker.track("change_in_mem", change_in_mem)
    assert result == 16
```

Which would produce this file in `tests/unit/unit_results.json`:
```json
{
  "exit_status": 0,
  "metrics": {
    "test_logger::test_exponentiate": {
      "result": 16,
      "memory_after_exponentiating": 551.0,
      "change_in_mem": 0.0,
      "_elapsed": 0.1190798282623291
    }
  },
  "gpu_types": [
    "NVIDIA H100 80GB HBM3"
  ],
  "coverage": 24.55897613282601
}
```

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
