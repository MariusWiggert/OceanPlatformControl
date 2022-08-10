"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

def timer(func):
    import time
    def wrapper():
        start = time.time()
        func()
        print(f"Time taken: {time.time()-start} seconds.")
    return wrapper


def convert_degree_to_km(degrees: List[float]) -> List[float]:
    pass