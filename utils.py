import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.5f} s'.format(name,(time.time() - t0)))