import os 
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def get_dirs():
    path = os.path.dirname(os.path.abspath(__file__))

    alex_pics = [f"{path}/data/Alex/{fname}" for fname in os.listdir(f"{path}/data/Alex")]
    alabel = [0] * len(alex_pics)

    kelly_pics = [f"{path}/data/Kelly/{fname}" for fname in os.listdir(f"{path}/data/Kelly")]
    klabel = [1] * len(kelly_pics)

    df = pd.DataFrame({"label": alabel + klabel, "path" : alex_pics + kelly_pics})
    return df

def read_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

if __name__ == "__main__":
    df = get_dirs()
    df["image"] = df["path"].map(read_image)

    image_array = df["image"].iloc[0]

    plt.imshow(image_array)
    plt.show()
