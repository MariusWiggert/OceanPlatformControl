import sys

from ocean_navigation_simulator.ocean_observer.Other.script_convert_data import main

i = 0
for ds in ["small", "medium", "big"]:
    for tile in ["small", "medium", "big"]:
        # to run on the slow gpu machine
        # print(
        #     f"scp -i ~/.ssh/azure-NN.pem -r ./training_{ds}_ds_CNN_{tile}_tile killian2k@fastnn.eastus.cloudapp.azure.com:/home/killian2k/seaweed/OceanPlatformControl/data_ablation_study/")
        # print(
        #     f"scp -i ~/.ssh/azure-NN.pem -r ./validation_{ds}_ds_CNN_{tile}_tile killian2k@fastnn.eastus.cloudapp.azure.com:/home/killian2k/seaweed/OceanPlatformControl/data_ablation_study/")
        # print(
        #     f"scp -i ~/.ssh/azure-NN.pem -r killian2k@fastnn.eastus.cloudapp.azure.com:/home/killian2k/seaweed/OceanPlatformControl/data_ablation_study/training_{ds}_ds_CNN_{tile}_tile ./data_ablation_study")
        # print(
        #     f"scp -i ~/.ssh/azure-NN.pem -r killian2k@fastnn.eastus.cloudapp.azure.com:/home/killian2k/seaweed/OceanPlatformControl/data_ablation_study/validation_{ds}_ds_CNN_{tile}_tile ./data_ablation_study")
        if tile != "big":
            continue
        print(f"creating CNN_dataset {i} {ds} {tile}")
        i += 1
        # if i > 3:
        path = f"./ablation_study/configs_NN/CNN_{ds}_ds/CNN_{ds}_ds_{tile}_tile"
        name_file = "ocean_navigation_simulator/ocean_observer/models/OceanCurrentRunner.py"
        # name_file = "scripts/test_scripts/generate_problems.py"
        sys.argv = (name_file + " " + f"--file-configs={path}").split(" ")
        main()

print("over.")
