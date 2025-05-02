import contextlib
import io

try:
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        # Silence the logging because NeMo is very verbose
        from nemo.tron.init import initialize_megatron  # noqa: F401
    INSTALLED = True
except ImportError:
    INSTALLED = False
print(f"NeMo {INSTALLED=}")
