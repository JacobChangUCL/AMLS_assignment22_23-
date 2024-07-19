import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


# 定义数据集类
class CelebADataset(Dataset):
    def __init__(self, image_folder, labels_file, transform=None):
        self.image_folder = image_folder
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0].split('\t')[1].strip())
        image = Image.open(img_name).convert('RGB')
        # the labels for the celeba dataset.
        # The first column is the index.
        # The second column is the corresponding file name.
        # The third column is the gender ({-1, +1}).
        # The last column is whether the person is smiling or not smiling ({-1, +1})
        label = int(self.labels.iloc[idx, 0].split('\t')[3].strip()) > 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(int(label))


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化数据集
train_dataset = CelebADataset(image_folder='../Datasets/celeba/img', labels_file='../Datasets/celeba/labels.csv',
                              transform=transform)
test_dataset = CelebADataset(image_folder='../Datasets/celeba_test/img', labels_file='../Datasets/celeba_test/labels.csv',
                             transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class GenderClassificationModel(nn.Module):
    def __init__(self):
        super(GenderClassificationModel, self).__init__()
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features.fc = nn.Linear(self.features.fc.in_features, 2)

    def forward(self, x):
        x = self.features(x)
        return x


model = GenderClassificationModel()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


train_model(model, train_loader, criterion, optimizer)


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'gender_detection_model.pth')

if __name__ == "__main__":
    # 加载模型
    model = GenderClassificationModel()
    model.load_state_dict(torch.load('gender_detection_model.pth'))
    model = model.to(device)

    # 进行预测
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print(f'Predicted: {predicted.cpu().numpy()}, Actual: {labels.cpu().numpy()}')
            break  # 仅打印一批次结果
