"""
Generator class to generate pair indices on the fly so all indices do not have 
to be store in memory.
"""

from collections.abc import Generator
import numpy as np
import numba as nb


class IndexPairGenerator(Generator):
    def __init__(self, n: int, chunk_size: int):
        self.n = n
        self.cur = 0
        self.chunk_size = chunk_size

    def send(self, value):
        indices, self.cur, self.n = IndexPairGenerator.get_part_problem(self.cur, self.n, self.chunk_size)
        if len(indices) == 0:
            self.throw()
        return np.array(indices)

    def throw(self, typ=None, val=None, tb=None):
        # raise StopIteration
        return []

    @staticmethod
    @nb.jit(nopython=True)
    def get_part_problem(cur, n, chunk_size):
        i = []
        j = []
        while len(i) < chunk_size:
            length = n - cur -1
            if length == 0:
                break
            i.extend(np.full(n-cur-1, cur))
            j.extend(range(cur+1, n))
            cur += 1
        indices = [np.array(i),np.array(j)]
        return indices, cur, n


def timer(func):
    import time
    def wrapper():
        start = time.time()
        func()
        print(f"Time taken: {time.time()-start} seconds.")
    return wrapper


@timer
def main():
    MAX_NUM_PAIRS = 1000000
    n = 100000
    gen = IndexPairGenerator(n, MAX_NUM_PAIRS)
    while True:
        val = next(gen)
        if len(val[0]) == 0:
            break
        # print(f"{val}\n")


if __name__ == "__main__":
    main()
