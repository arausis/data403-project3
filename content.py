import os 
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


path = os.path.dirname(os.path.abspath(__file__))

def get_dirs():

    alex_files = os.listdir(f"{path}/data/Alex")
    alex_pics = [f"{path}/data/Alex/{fname}" for fname in alex_files]
    alabel = [0] * len(alex_pics)

    kelly_files = os.listdir(f"{path}/data/Kelly")
    kelly_pics = [f"{path}/data/Kelly/{fname}" for fname in kelly_files]
    klabel = [1] * len(kelly_pics)

    df = pd.DataFrame({"fname" : alex_files + kelly_files, "label": alabel + klabel, "path" : alex_pics + kelly_pics})
    return df

def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    return image_array

def get_image_data():
    df = get_dirs()
    df["image"] = df["path"].map(read_image)

    manual_tagging = Path(f"{path}/manual_tagging_content.csv")

    if manual_tagging.is_file():
        df_tags = pd.read_csv(f"{path}/manual_tagging_content.csv")

        df = pd.merge(df, df_tags, on="fname")
    return df

# A function to let us manually tag images with certain features. Results are stored in a .csv and are 
def tag_images():
    df = get_image_data()

    # event = concert/game/sports night
    # bodwinPeople = blonde girl, husband, stats profs

    manual_columns = ["person", "building", "indoors", "bodwinPeople", "event", "gameNight", "sports", "concert"] 


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



        done = int(input(f"0/1 continue?: "))
        if(done == 1):
            manual_input.append(input_map)
        else:
            input_map = {}
            for col in manual_columns:
                input_map[col] = int(input(f"0/1 | {col} : "))
                manual_input.append(input_map)
        # manual_input.append(input_map)

        plt.close()
    
    # Reformat the inputs so that we can make a dataframe
    df_dict = {}
    df_dict["fname"] = list(df["fname"])

    for col in manual_columns:
        df_dict[col] = [x[col] for x in manual_input]

    input_df = pd.DataFrame(df_dict)
    input_df.to_csv(f"{path}/manual_tagging_content.csv")


if __name__ == "__main__":
    tag_images()
