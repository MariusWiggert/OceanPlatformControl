from collections import Counter
import socket
import time
import ray
from requests import get
import os

# ray.init(address='auto')
# ray.init("ray://13.68.187.126:10001")
ray.init("ray://localhost:10001")


# ray up setup/ray-config.yaml
# ray monitor setup/ray-config.yaml
# ray dashboard setup/ray-config.yaml
# ray submit setup/ray-config.yaml scripts/jerome/3_Cluster_RL/evaluate_controller_ray.py
# ray attach setup/ray-config.yaml -p 10001

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU'] if 'CPU' in ray.cluster_resources() else 0))

# print('''This cluster consists of
#     {} nodes in total'''.format(len(ray.nodes())))
#
# print(ray.cluster_resources())
# print(ray.nodes())

@ray.remote(num_cpus=1)
def f():
    time.sleep(1)
    ip = get('https://api.ipify.org').content.decode('utf8')
    pid = os.getpid()

    print(f'Task Finished ({ip}, {pid})')

    # print('My public IP address is: {}'.format(ip))

    # Return IP address.
    # return socket.gethostbyname(socket.gethostname())
    return f'{ip}, {pid}'

object_ids = [f.remote() for _ in range(100)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')

for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))