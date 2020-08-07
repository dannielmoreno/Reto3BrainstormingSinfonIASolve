## Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torchsummary import summary
## Check sizes of labeled and unlabeled data

XTrain = pd.read_csv("DataTrain.csv")
YTrain = pd.read_csv("LabelsTrain.csv")
XTest = pd.read_csv("DataTest.csv")
print(YTrain.shape)
print(XTrain.shape)
print(XTest.shape)
## Divide into train, dev and test sets

Xtrain = pd.read_csv("DataTrain.csv")[:14000]
Xdev = pd.read_csv("DataTrain.csv")[14000:]
Ytrain = pd.read_csv("LabelsTrain.csv")[:14000]
Ydev = pd.read_csv("LabelsTrain.csv")[14000:]
Xtest = pd.read_csv("DataTest.csv")
print("Xtrain: ", Xtrain.shape)
print("Ytrain: ", Ytrain.shape)
print("Xdev: ", Xdev.shape)
print("Ydev: ", Ydev.shape)
print("Xtest: ", Ydev.shape)

## Creating classes of datasets and a tensor for the test without labels

class SinfoniaTrainDataset(Dataset):
    def __init__(self):
        self.x = torch.FloatTensor(Xtrain.values)
        self.y = torch.FloatTensor(Ytrain.values)
        self.n_samples = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples

class SinfoniaDevDataset(Dataset):
    def __init__(self):
        self.x = torch.FloatTensor(Xdev.values)
        self.y = torch.FloatTensor(Ydev.values)
        self.n_samples = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples


Xtest = torch.FloatTensor(Xtest.values)

## Creating classes of models to validate

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 24)
        self.fc4 = nn.Linear(24, 12)
        self.fc5 = nn.Linear(12, 6)
        self.fc6 = nn.Linear(6, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x

class Model2(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(64, 24)
        self.bn3 = nn.BatchNorm1d(24)
        self.dropout3 = nn.Dropout(p=p)
        self.fc4 = nn.Linear(24, 12)
        self.bn4 = nn.BatchNorm1d(12)
        self.dropout4 = nn.Dropout(p=p)
        self.fc5 = nn.Linear(12, 6)
        self.bn5 = nn.BatchNorm1d(6)
        self.dropout5 = nn.Dropout(p=p)
        self.fc6 = nn.Linear(6, 1)
    def forward(self, x):
        x = self.bn1(self.dropout1(F.relu(self.fc1(x))))
        x = self.bn2(self.dropout2(F.relu(self.fc2(x))))
        x = self.bn3(self.dropout3(F.relu(self.fc3(x))))
        x = self.bn4(self.dropout4(F.relu(self.fc4(x))))
        x = self.bn5(self.dropout5(F.relu(self.fc5(x))))
        x = F.sigmoid(self.fc6(x))
        return x

class Model3(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(32, 64)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(64, 24)
        self.dropout3 = nn.Dropout(p=p)
        self.fc4 = nn.Linear(24, 12)
        self.dropout4 = nn.Dropout(p=p)
        self.fc5 = nn.Linear(12, 6)
        self.dropout5 = nn.Dropout(p=p)
        self.fc6 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        x = F.sigmoid(self.fc6(x))
        return x

class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x

class Model5(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.dropout1 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(32, 64)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(64, 128)
        self.dropout3 = nn.Dropout(p=p)
        self.fc4 = nn.Linear(128, 256)
        self.dropout4 = nn.Dropout(p=p)
        self.fc5 = nn.Linear(256, 512)
        self.dropout5 = nn.Dropout(p=p)
        self.fc6 = nn.Linear(512, 1)
    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        x = F.sigmoid(self.fc6(x))
        return x

class Model6(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 64)
        self.fc7 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.sigmoid(self.fc7(x))
        return x

class Model7(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 64)
        self.fc8 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.sigmoid(self.fc8(x))
        return x

class Model8(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.fc2 = nn.Linear(32, 64)
        self.dropout2 = nn.Dropout(p=p)
        self.fc3 = nn.Linear(64, 128)
        self.dropout3 = nn.Dropout(p=p)
        self.fc4 = nn.Linear(128, 256)
        self.dropout4 = nn.Dropout(p=p)
        self.fc5 = nn.Linear(256, 512)
        self.dropout5 = nn.Dropout(p=p)
        self.fc6 = nn.Linear(512, 64)
        self.dropout6 = nn.Dropout(p=p)
        self.fc7 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        x = self.dropout5(F.relu(self.fc5(x)))
        x = self.dropout6(F.relu(self.fc6(x)))
        x = F.sigmoid(self.fc7(x))
        return x

## Creating data loaders

trainset = SinfoniaTrainDataset()
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
devset = SinfoniaDevDataset()
devloader = DataLoader(devset)

## Training an epoch, validate and main functions

def train_epoch(model, optimizer, criterion):
    LOSS = 0
    for data in trainloader:
        X, y = data
        model.zero_grad()
        output = model(X.view(-1, 28))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        LOSS += loss
    LOSS = LOSS / len(trainloader)
    return LOSS.item()

def validate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            X, y = data
            output = torch.round(model(X.view(-1, 28)))
            for i in range(len(output)):
                if output[i] == y[i]:
                    correct += 1
                total += 1
    accuracy = round(correct / total, 5)
    return accuracy, correct, total

def main(model, learning_rate=0.0001, EPOCHS=100, criterion = nn.BCELoss()):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_array = []
    accuracy_train_array = []
    accuracy_dev_array = []
    max_accuracy = 0
    max_accuracy_epoch = 0
    for epoch in range(1, EPOCHS+1):
        print("Epoch " + str(epoch))
        model.train()
        loss_of_epoch = train_epoch(model, optimizer, criterion)
        loss_array.append(loss_of_epoch)
        model.eval()
        accuracyTrain, correctTrain, totalTrain = validate(model, trainloader)
        accuracyDev, correctDev, totalDev = validate(model, devloader)
        accuracy_train_array.append(accuracyTrain)
        accuracy_dev_array.append(accuracyDev)
        print("TRAINSET - Accuracy: ", accuracyTrain, " Correct: ", correctTrain, " Total: ", totalTrain)
        print("DEVSET   - Accuracy: ", accuracyDev, " Correct: ", correctDev, " Total: ", totalDev)
        if accuracyDev >= max_accuracy:
            best_model_dict = model.state_dict()
            max_accuracy = accuracyDev
            max_accuracy_epoch = epoch


    return loss_array, accuracy_train_array, accuracy_dev_array, max_accuracy, max_accuracy_epoch, best_model_dict


##
def graph_stats(name_of_model, loss_array, accuracy_train_array, accuracy_dev_array, max_accuracy, max_accuracy_epoch):
    EPOCHS = len(accuracy_train_array)
    plt.figure()
    plt.title(name_of_model + ": Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, EPOCHS+1), loss_array)
    plt.figure()
    plt.title(name_of_model + ": Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(1, EPOCHS + 1), accuracy_train_array, "b", label="Train set")
    plt.plot(range(1, EPOCHS + 1), accuracy_dev_array, "r", label="Dev set")
    plt.plot(max_accuracy_epoch, max_accuracy, "og", label="Best Model Performance in Dev set")
    plt.legend()

##
def save_model(name_of_model, loss_array, accuracy_train_array, accuracy_dev_array, max_accuracy, max_accuracy_epoch, best_model_dict):
    torch.save(best_model_dict, "best_model_" + name_of_model + ".pth")
    with open("loss_array_" + name_of_model + ".txt", "wb") as f:
        pickle.dump(loss_array, f)
    with open("accuracy_train_array_" + name_of_model + ".txt", "wb") as f:
        pickle.dump(accuracy_train_array, f)
    with open("accuracy_dev_array_" + name_of_model + ".txt", "wb") as f:
        pickle.dump(accuracy_dev_array, f)
    with open("max_accuracy_" + name_of_model + ".txt", "w") as f:
        f.write(str(max_accuracy))
    with open("max_accuracy_epoch_" + name_of_model + ".txt", "w") as f:
        f.write(str(max_accuracy_epoch))

##
def load_model(name_of_model):
    with open("loss_array_" + name_of_model + ".txt", "rb") as f:
        loaded_loss_array = pickle.load(f)
    with open("accuracy_train_array_" + name_of_model + ".txt", "rb") as f:
        loaded_accuracy_train_array = pickle.load(f)
    with open("accuracy_dev_array_" + name_of_model + ".txt", "rb") as f:
        loaded_accuracy_dev_array = pickle.load(f)
    with open("max_accuracy_" + name_of_model + ".txt", "r") as f:
        loaded_max_accuracy = float(f.read())
    with open("max_accuracy_epoch_" + name_of_model + ".txt", "r") as f:
        loaded_max_accuracy_epoch = float(f.read())
    return name_of_model, loaded_loss_array, loaded_accuracy_train_array, loaded_accuracy_dev_array, loaded_max_accuracy, loaded_max_accuracy_epoch

## Train and set various models

model6 = Model6()
save_model("model6", *main(model6))

##
model7 = Model7()
save_model("model7", *main(model7))
##
model8_d02 = Model8(0.2)
save_model("model8_d02", *main(model8_d02))

##
model8_d01 = Model8(0.1)
save_model("model8_d01", *main(model8_d01))

## Graph the statistics of the different pretrained models

graph_stats(*load_model("model6"))
graph_stats(*load_model("model7"))
graph_stats(*load_model("model8_d02"))
graph_stats(*load_model("model8_d01"))

## Make an inference with the final model that you choose

finalmodel = Model8(0.1)
finalmodel.load_state_dict(torch.load("best_model_model8_d01.pth"))

finalmodel.eval()

def inference_with_test(finalmodel):
    finalmodel.eval()
    with torch.no_grad():
        output = torch.round(finalmodel(Xtest.view(-1, 28)))
    with open('reto3_201913389.txt', 'w') as f:
        for tag in output:
            f.write("%s\n" % int(tag.item()))

inference_with_test(finalmodel)

print(finalmodel)
summary(finalmodel, (1,28))









