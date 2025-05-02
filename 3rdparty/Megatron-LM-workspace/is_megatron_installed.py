try:
    from megatron.core import parallel_state  # noqa: F401

    INSTALLED = True
except ImportError:
    INSTALLED = False

print(f"Megatron {INSTALLED=}")
