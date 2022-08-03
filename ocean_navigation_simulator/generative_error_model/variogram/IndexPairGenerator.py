"""
Generator class to generate pair indices on the fly so all indices do not have 
to be store in memory.
"""

from collections.abc import Generator
import itertools
import numpy as np

class IndexPairGenerator(Generator):
    def __init__(self, n: int, chunk_size: int):
        self.n = n
        self.cur = 0
        self.chunk_size = chunk_size

    def send(self, value):
        indices = self.get_part_problem()
        if len(indices) == 0:
            self.throw()
        return np.array(indices)

    def throw(self, typ=None, val=None, tb=None):
        # raise StopIteration
        return []

    def get_part_problem(self):
        i = []
        j = []
        while len(i) < self.chunk_size:
            length = self.n - self.cur -1
            if length == 0:
                break
            i.extend(np.full(self.n-self.cur-1, self.cur))
            j.extend(range(self.cur+1, self.n))
            self.cur += 1
        indices = [np.array(i),np.array(j)]
        return indices


if __name__ == "__main__":
    # test generator
    MAX_NUM_PAIRS = 3
    n = 5
    gen = IndexPairGenerator(n)
    while True:
        val = next(gen)
        if len(val[0]) == 0:
            break
        print(f"{val}\n")
