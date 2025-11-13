from contextlib import contextmanager
import tempfile
import os

def format_number(x, digits=1):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

@contextmanager
def bytes_to_tempfile(b: bytes, suffix: str = ""):
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b)
        yield path
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
