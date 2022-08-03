"""
    Ray clogs up the ~/ray_results directory by creating folders for every training start, even when
    canceling after a few iterations. This script removes all short trainings in order to simplify
    finding the important trainings in tensorboard. It however ignores the very last experiment,
    since it could be still ongoing.
"""

import os
import shutil
from pathlib import Path

def run(iteration_limit, delete, filter_string='', ignore_last=True):
    ray_results = str(Path.home()) + '/ray_results'
    experiments = os.listdir(ray_results)
    experiments = [x for x in experiments if not x.startswith('.')]
    if filter_string != '':
        experiments = [x for x in experiments if x.startswith(filter_string)]

    experiments = [os.path.join(ray_results, f) for f in experiments]
    experiments.sort(key=lambda x: os.path.getmtime(x))

    # don't touch the last experiment, it might still being created
    for experiment in experiments[:-1] if ignore_last else experiments:
        csv_file = experiment+'/progress.csv'
        if os.path.isfile(csv_file):
            with open(csv_file) as f:
                row_count = sum(1 for line in f)
                if row_count < iteration_limit:
                    if delete:
                        shutil.rmtree(experiment)
                    print(f'Delete:  {csv_file} with {row_count} rows')
                # else:
                #     print(f'Keep:  {csv_file} with {row_count} rows')
        else:
            if delete:
                shutil.rmtree(experiment)
            print(f'Delete:  {csv_file} without csv file')

    # print(f'Ignore: {experiments[-1]} with {sum(1 for line in open(experiments[-1]+"/progress.csv"))} rows')

if __name__ == "__main__":
    run(iteration_limit= 10, delete= True)