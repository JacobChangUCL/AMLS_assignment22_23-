import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models


# Specify the dataset for the task
specify_dataset_for_task = {
    "A1": {
        "task": "gender_detection",
        "train_image_folder": "./Datasets/celeba/img",
        "train_labels_file": "./Datasets/celeba/labels.csv",
        "test_image_folder": "./Datasets/celeba_test/img",
        "test_labels_file": "./Datasets/celeba_test/labels.csv",
        "num_classes": 2,
    },
    "A2": {
        "task": "emotion_detection",
        "train_image_folder": "./Datasets/celeba/img",
        "train_labels_file": "./Datasets/celeba/labels.csv",
        "test_image_folder": "./Datasets/celeba_test/img",
        "test_labels_file": "./Datasets/celeba_test/labels.csv",
        "num_classes": 2,
    },
    "B1": {
        "task": "face_shape_recognition",
        "train_image_folder": "./Datasets/cartoon_set/img",
        "train_labels_file": "./Datasets/cartoon_set/labels.csv",
        "test_image_folder": "./Datasets/cartoon_set_test/img",
        "test_labels_file": "./Datasets/cartoon_set_test/labels.csv",
        "num_classes": 5,
    },
    "B2": {
        "task": "eye_color_recognition",
        "train_image_folder": "./Datasets/cartoon_set/img",
        "train_labels_file": "./Datasets/cartoon_set/labels.csv",
        "test_image_folder": "./Datasets/cartoon_set_test/img",
        "test_labels_file": "./Datasets/cartoon_set_test/labels.csv",
        "num_classes": 5,
    },
    "batch_size": 32,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "criterion": nn.CrossEntropyLoss(),
    "layers" : [50,101,152],  #18,34,50,101,152
    "lr" : 0.001,
    "num_epochs" : 10,
    "num_rounds" : 10
}


# Define the dataset class
class AMLS_Dataset(Dataset):
    def __init__(self, task_name, image_folder, labels_file, transform=None):
        self.task_name = task_name
        self.image_folder = image_folder
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # the labels for the celeba dataset.
        # The first column is the index.
        # The second column is the corresponding file name.
        # The third column is the gender ({-1, +1}).
        # The last column is whether the person is smiling or not smiling ({-1, +1})
        # ******************************************************************
        # the labels for the cartoon set.
        # The first column is the index.
        # The second column is eye colour (0-4),
        # the third column is face shape (0-4),
        # the last column is the corresponding file name.
        if specify_dataset_for_task[self.task_name]["task"] == "gender_detection":
            img_name = os.path.join(
                self.image_folder, self.labels.iloc[idx, 0].split("\t")[1].strip()
            )
            label = int(self.labels.iloc[idx, 0].split("\t")[2].strip()) > 0
            label = torch.tensor(int(label))
        elif specify_dataset_for_task[self.task_name]["task"] == "emotion_detection":
            img_name = os.path.join(
                self.image_folder, self.labels.iloc[idx, 0].split("\t")[1].strip()
            )
            label = int(self.labels.iloc[idx, 0].split("\t")[3].strip()) > 0
            label = torch.tensor(int(label))
        elif (
            specify_dataset_for_task[self.task_name]["task"] == "face_shape_recognition"
        ):
            img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0].split('\t')[-1].strip())
            label = int(self.labels.iloc[idx, 0].split("\t")[2].strip())
            label = torch.tensor(label)
        elif (
            specify_dataset_for_task[self.task_name]["task"] == "eye_color_recognition"
        ):
            img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0].split('\t')[-1].strip())
            label = int(self.labels.iloc[idx, 0].split("\t")[1].strip())
            label = torch.tensor(label)
        else:
            raise ValueError("Task assignment error")

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Data preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Define the model
class AMLS_Model(nn.Module):
    def __init__(self, num_classes, layer_number):
        super(AMLS_Model, self).__init__()
        if layer_number == 18:
            self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif layer_number == 34:
            self.features = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif layer_number == 50:
            self.features = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        elif layer_number == 101:
            self.features = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        elif layer_number == 152:
            self.features = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("The number of layers in ResNet is wrong.")

        # Freeze the parameters of the features layer.
        # for param in self.features.parameters():
        #     param.requires_grad = False
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x


def get_optimizer(model, lr = specify_dataset_for_task["lr"]):
    return optim.Adam(model.parameters(), lr=lr)


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs = specify_dataset_for_task["num_epochs"]):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(specify_dataset_for_task["device"])
            labels = labels.to(specify_dataset_for_task["device"])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(specify_dataset_for_task["device"])
            labels = labels.to(specify_dataset_for_task["device"])

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy