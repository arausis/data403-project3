import pandas as pd
import matplotlib.pyplot as plt
from read_data import get_dirs, get_holdout
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

class PhotoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label


transform = transforms.Compose([
    # change image size
    transforms.Resize((224, 224)),
    # converts PIL image to PyTorch tensor
    transforms.ToTensor()
])


df = get_dirs()
df_holdout, df_training = get_holdout(df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)

holdout_dataset = PhotoDataset(df_holdout, transform=transform)

def evaluate_model(batch_size):
  # initialising every image transformed
  dataset = PhotoDataset(df_training, transform=transform)

  # data wrapper for convenient usage to model
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  # 20 images/labels, 3 colors, 224 height, 224 width
  # 20 at a time
  images, labels = next(iter(train_loader))
  print(images.shape, labels.shape)

  # resnet18 model with default weights (pre-trained IMAGENET1K_V1)
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  # X*W + c, using final layer nodes to output ???
  model.fc = nn.Linear(model.fc.in_features, 2)   # Alex vs Kelly
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()

  # optimizer on weights using learning rate 1e-4
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  batch_losses = []
  batch_accs = []

  model.train()

  for batch_idx, (imgs, lbls) in enumerate(tqdm(train_loader)):
      imgs, lbls = imgs.to(device), lbls.to(device)

      # clear old gradients
      optimizer.zero_grad()

      # output of model
      outputs = model(imgs)

      # calculate loss based on current output/weights
      loss = criterion(outputs, lbls)

      # backward propagation to train model
      loss.backward()

      # forward propagation to update parameters
      optimizer.step()

      preds = outputs.argmax(dim=1)
      acc = (preds == lbls).float().mean().item()

      batch_losses.append(loss.item())
      batch_accs.append(acc)
  
  y_true = []
  y_pred = []

  with torch.no_grad():
      for img, label in PhotoDataset(df_holdout, transform=transform):
          img = img.unsqueeze(0).to(device)   # add batch dimension: (3,224,224) → (1,3,224,224)
          out = model(img)
          pred = out.argmax(dim=1).item()

          y_true.append(label)
          y_pred.append(pred)
  acc = np.mean(np.array(y_true) == np.array(y_pred))

  print(f"Batchsize: {batch_size}, accuracy:{acc}")

batch_size_candidates = np.linspace(20, 30, 10)
for batch_size in batch_size_candidates:
  evaluate_model(int(batch_size))
