import ray

ray.init()


@ray.remote(num_cpus=2)
def function(row):
    pass


task = [function.remote(index) for index in range(10)]

ray_results = ray.get(task)
