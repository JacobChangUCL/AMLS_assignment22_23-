import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import AMLS_Model,AMLS_Dataset,specify_dataset_for_task,transform,train_model,evaluate_model,get_optimizer


task_name = "B2"
# Specify the dataset for the task
train_dataset = AMLS_Dataset(task_name = task_name,
                             image_folder = specify_dataset_for_task[task_name]["train_image_folder"],
                             labels_file = specify_dataset_for_task[task_name]["train_labels_file"],
                             transform=transform)
test_dataset = AMLS_Dataset(task_name = task_name,
                            image_folder = specify_dataset_for_task[task_name]["test_image_folder"],
                            labels_file = specify_dataset_for_task[task_name]["test_labels_file"],
                            transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size = specify_dataset_for_task["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = specify_dataset_for_task["batch_size"], shuffle=False)


if __name__ == "__main__":

    for lay in specify_dataset_for_task["layers"]:
        model = AMLS_Model(num_classes=specify_dataset_for_task[task_name]["num_classes"],layer_number=lay)

        # Use GPU
        model = model.to(specify_dataset_for_task["device"])
        criterion = specify_dataset_for_task["criterion"]
        optimizer = get_optimizer(model)

        test_accuracy = []
        for i in range(specify_dataset_for_task["num_rounds"]):
            print("ResNet-{}, round {}:".format(lay,i + 1))
            # Train the model
            train_model(model, train_loader, criterion, optimizer)
            test_accuracy.append(evaluate_model(model, test_loader))
            the_trained_model = "{}/{}_model{}_{}.pth".format(task_name,specify_dataset_for_task[task_name]["task"],lay,i)
            # Save the trained model
            torch.save(model.state_dict(), the_trained_model)


            # Load the model
            model = AMLS_Model(num_classes=specify_dataset_for_task[task_name]["num_classes"],layer_number=lay)
            model.load_state_dict(torch.load(the_trained_model))
            model = model.to(specify_dataset_for_task["device"])

            # Make predictions
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(specify_dataset_for_task["device"])
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    print(f'Predicted: {predicted.cpu().numpy()}, Actual: {labels.cpu().numpy()}')
                    break  # Print only one batch of predictions
            print("*" * 30)

        # Calculate the mean and standard deviation
        mean = np.mean(test_accuracy)
        std_dev = np.std(test_accuracy)

        # Format the output string
        formatted_output = f"{mean:.2f}%~{std_dev:.2f}%"
        print(formatted_output)
        print()