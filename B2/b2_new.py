import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# 定义数据集类
class CelebADataset(Dataset):
    def __init__(self, image_folder, labels_file, transform=None):
        self.image_folder = image_folder
        self.labels = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0].split('\t')[-1].strip())
        image = Image.open(img_name).convert('RGB')
        #the labels for the cartoon set.
        # The first column is the index.
        # The second column is eye colour (0-4),
        # the third column is face shape (0-4),
        # the last column is the corresponding file name.
        label = int(self.labels.iloc[idx, 0].split('\t')[1].strip())

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化数据集
train_dataset = CelebADataset(image_folder='../Datasets/cartoon_set/img', labels_file='../Datasets/cartoon_set/labels.csv',
                              transform=transform)
test_dataset = CelebADataset(image_folder='../Datasets/cartoon_set_test/img', labels_file='../Datasets/cartoon_set_test/labels.csv',
                             transform=transform)

train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class EyeColorRecognitionModel(nn.Module):
    def __init__(self):
        super(EyeColorRecognitionModel, self).__init__()
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features.fc = nn.Linear(self.features.fc.in_features, 5)

    def forward(self, x):
        x = self.features(x)
        return x


model = EyeColorRecognitionModel()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 早停策略
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(patience=5, delta=0.001)


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
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

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


train_model(model, train_loader, val_loader, criterion, optimizer)


# 评估模型
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('checkpoint.pth'))
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
torch.save(model.state_dict(), 'eye_color_recognition_model.pth')

if __name__ == "__main__":
    # 加载模型
    model = EyeColorRecognitionModel()
    model.load_state_dict(torch.load('eye_color_recognition_model.pth'))
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