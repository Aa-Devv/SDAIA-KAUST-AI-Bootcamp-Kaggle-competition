import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt






train_df = pd.read_csv("train.csv", dtype="uint8")
test_df = pd.read_csv("test.csv", dtype="uint8")


train_features = train_df.drop("labels", axis=1)
train_target = train_df.loc[:, "labels"]
test_features = pd.read_csv("test.csv")


class DataSetWithTransforms(Dataset):

    def __init__(self, features, target, feature_transforms=None):
        super().__init__()
        self._features = features
        self._target = torch.from_numpy(target).long()
        self._feature_transforms = feature_transforms

    def __getitem__(self, index):
        if self._feature_transforms is None:
            features = self._features[index]
            # feature = torch.rashape(feature, (32,32,3))
        else:
            features = self._feature_transforms(self._features[index])
        target = self._target[index]
        return (features, target)
    def __len__(self):
        n_samples, _ = self._features.shape
        return n_samples

class DataSetTest(Dataset):

     def __init__(self, features, feature_transforms=None):
         super().__init__()
         self._features = features
         self._feature_transforms = feature_transforms

     def __getitem__(self, index):
         if self._feature_transforms is None:
             features = self._features[index]
             # feature = torch.rashape(feature, (32,32,3))
         else:
             features = self._feature_transforms(self._features[index])
         return (features)

     def __len__(self):
         n_samples, _ = self._features.shape
         return n_samples

     # data augmentation should only apply to training data
_feature_transforms = transforms.Compose([
    transforms.Lambda(lambda array: array.reshape((32, 32))),
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=15, scale=(1.0, 1.1)),
    transforms.ToTensor(),
])

train_dataset = DataSetWithTransforms(train_features.values, train_target.values, _feature_transforms)


train_loader = DataLoader(train_dataset, 4)


_feature_transforms_test = transforms.Compose([
    transforms.Lambda(lambda array: array.reshape((32, 32))),
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

test_dataset = DataSetTest(test_df.values, _feature_transforms_test)


test_loader=DataLoader(test_dataset,4)



class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.l1 = nn.Linear(5 * 5 * 50, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 29)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, 5 * 5 * 50)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x



net = ConvNet()
lr = .1
momentum = 0.0

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

ls = []
for i in range(100):
    loss_total = 0
    loss_val = 0
    acc_train = 0
    total_train = 0
    ke = []
    for ii, batch in enumerate(train_loader):
        data = batch[0]
        label = batch[1]

        optimizer.zero_grad()
        logits = net(data)
        ke.append(logits)
        ke.append(label)
        loss = F.cross_entropy(logits, label)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()
        out = torch.argmax(logits, dim=1)
        acc_train += torch.sum(out == label)
        total_train += logits.shape[0]
    ls.append(loss_total)
    lr_scheduler.step()
    print(f"Iteataion {i}: Training Accuracy: {acc_train.item() / total_train}")

plt.plot(ls)
plt.show(block=True)
test_loader = DataLoader(test_dataset, 64)

predictions_list = list()
for images in test_loader:

    predicition = net(images).argmax(1)
    predictions_list.append(predicition)

predictions_list = torch.cat(predictions_list).cpu()
test_features = pd.read_csv( "test.csv")
_ = (pd.DataFrame
     .from_dict({"Id": test_features.index, "Category": predictions_list})
     .to_csv("submission.csv", index=False))
