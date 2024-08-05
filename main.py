import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import AMLS_Model, AMLS_Dataset, specify_dataset_for_task, transform, evaluate_model

for task_name in ["A1", "A2", "B1", "B2"]:  # ,"A2","B1","B2"
    # Specify the dataset for the task
    test_dataset = AMLS_Dataset(task_name=task_name,
                                image_folder=specify_dataset_for_task[task_name]["test_image_folder"],
                                labels_file=specify_dataset_for_task[task_name]["test_labels_file"],
                                transform=transform)
    # Create data loaders,use mini-batch gradient descent
    test_loader = DataLoader(test_dataset, batch_size=specify_dataset_for_task["batch_size"], shuffle=False)

    # model selection stage
    for lay in specify_dataset_for_task["layers"]:  # ResNet [50,101,152]
        test_accuracy = []
        print("({}){}[ResNet-{}]--->>>".format(task_name, specify_dataset_for_task[task_name]["task"], lay))
        for i in range(10):
            # Load the model.
            model = AMLS_Model(num_classes=specify_dataset_for_task[task_name]["num_classes"], layer_number=lay)
            the_trained_model = "{}/{}_model{}_{}.pth".format(task_name, specify_dataset_for_task[task_name]["task"],
                                                              lay, i)
            model.load_state_dict(torch.load(the_trained_model))
            model = model.to(specify_dataset_for_task["device"])
            test_accuracy.append(evaluate_model(model, test_loader))

        # Calculate the mean and standard deviation
        mean = np.mean(test_accuracy)
        std_dev = np.std(test_accuracy)

        # Format the output string
        print("({}){}[ResNet-{}]:{:.2f}%~{:.2f}%".format(task_name, specify_dataset_for_task[task_name]["task"], lay,
                                                         mean, std_dev))
        print()
