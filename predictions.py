import pandas as pd
import matplotlib.pyplot as plt
from read_data import *
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

        return image, 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 # resnet18 model with default weights (pre-trained IMAGENET1K_V1)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  # X*W + c, using final layer nodes to output ???
model.fc = nn.Linear(model.fc.in_features, 2)   # Alex vs Kelly
model = model.to(device)

model.load_state_dict(torch.load("final_model_full.pth"))
model.eval()


# Load our test data
transform = transforms.Compose([
    # change image size
    transforms.Resize((224, 224)),
    # converts PIL image to PyTorch tensor
    transforms.ToTensor()
])

df = get_test_data()
holdout_dataset = PhotoDataset(df, transform=transform)
loader = DataLoader(holdout_dataset, batch_size=32, shuffle=False)

all_preds = []
all_probs = []

with torch.no_grad():
    for imgs, _ in loader:
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

df["pred"] = all_preds
df["probabilities"] = all_probs

df["pred"] = df["pred"].map(lambda x: "Kelly" if x == 1 else "Alex")

print(df[["fname", "pred", "probabilities"]])
