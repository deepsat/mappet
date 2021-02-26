import contextlib
import time


@contextlib.contextmanager
def timed_log(log, text):
    start = time.time()
    yield
    end = time.time()
    log(text.format(time=end-start))
