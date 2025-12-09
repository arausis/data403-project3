import os 
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


path = os.path.dirname(os.path.abspath(__file__))

holdout_map = {
    "TestSetImage01.png" : "Kelly",
    "TestSetImage02.png" : "Alex",
    "TestSetImage03.png" : "Alex",
    "TestSetImage04.png" : "Alex",
    "TestSetImage05.png" : "Alex",
    "TestSetImage06.png" : "Alex",
    "TestSetImage07.png" : "Kelly",
    "TestSetImage08.png" : "Kelly",
    "TestSetImage09.png" : "Kelly",
    "TestSetImage10.png" : "Kelly",
    "TestSetImage11.png" : "Kelly",
    "TestSetImage12.png" : "Alex",
    "TestSetImage13.png" : "Kelly",
    "TestSetImage14.png" : "Kelly",
    "TestSetImage15.png" : "Kelly",
    "TestSetImage16.png" : "Kelly",
    "TestSetImage17.png" : "Alex",
    "TestSetImage18.png" : "Alex",
    "TestSetImage19.png" : "Alex",
    "TestSetImage20.png" : "Alex"
}

def get_holdout(df):

    df_holdout = df.sample(frac=0.2, random_state=42)
    df_training  = df.drop(df_holdout.index)
    return df_holdout, df_training

def get_dirs():

    alex_files = os.listdir(f"{path}/data/Alex")
    alex_pics = [f"{path}/data/Alex/{fname}" for fname in alex_files]
    alabel = [0] * len(alex_pics)

    kelly_files = os.listdir(f"{path}/data/Kelly")
    kelly_pics = [f"{path}/data/Kelly/{fname}" for fname in kelly_files]
    klabel = [1] * len(kelly_pics)

    first_holdout = os.listdir(f"{path}/data/HoldoutSet01")
    holdout_pics = [f"{path}/data/HoldoutSet01/{fname}" for fname in first_holdout]
    holdout_labels = [holdout_map[x] for x in first_holdout]

    df = pd.DataFrame({"fname" : alex_files + kelly_files + first_holdout, "label": alabel + klabel + holdout_labels, "path" : alex_pics + kelly_pics + holdout_pics})
    return df

def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    return image_array

def get_image_data():
    df = get_dirs()
    df["image"] = df["path"].map(read_image)

    manual_tagging = Path(f"{path}/manual_tagging.csv")

    if manual_tagging.is_file():
        df_tags = pd.read_csv(f"{path}/manual_tagging.csv")

        df = pd.merge(df, df_tags, on="fname")
    return df

# A function to let us manually tag images with certain features. Results are stored in a .csv and are 
def tag_images():
    df = get_image_data()

    manual_columns = ["q1", "q2", "q3", "q4"]

    manual_input = []

    for file in list(df["fname"]):
        image_array = df[df["fname"].map(lambda x: x == file)]["image"].iloc[0]

        plt.imshow(image_array)
        plt.show(block=False)

        input_map = {}

        # Record the input for each column
        print()
        for col in manual_columns:
            input_map[col] = int(input(f"Y/n | {col} : ").strip().lower() == "y")

        manual_input.append(input_map)

        plt.close()
    
    # Reformat the inputs so that we can make a dataframe
    df_dict = {}
    df_dict["fname"] = list(df["fname"])

    for col in manual_columns:
        df_dict[col] = [x[col] for x in manual_input]

    input_df = pd.DataFrame(df_dict)
    input_df.to_csv(f"{path}/manual_tagging.csv")


if __name__ == "__main__":
    df = get_image_data()
    print(df.shape)

