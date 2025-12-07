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
    """
    Manually tag each image with binary (0/1) content features.
    - Saves to content_features.csv
    - Saves progress after each image
    - Skips images that are already tagged
    - After answering all prompts for an image, you can confirm or redo
      that image's answers before they are saved.
    """
    df = get_dirs()
    manual_columns = [
        "person", "building", "indoors", "bodwinPeople",
        "event", "gameNight", "sports", "concert"
    ]

    csv_path = Path(f"{path}/content_features.csv")

    # Load existing progress if available
    if csv_path.is_file():
        tagged_df = pd.read_csv(csv_path)
        tagged_files = set(tagged_df["fname"].tolist())
        print(f"Loaded {len(tagged_files)} previously tagged images.")
    else:
        tagged_df = pd.DataFrame(columns=["fname"] + manual_columns)
        tagged_files = set()

    for _, row in df.iterrows():
        fname = row["fname"]

        # Skip images already tagged
        if fname in tagged_files:
            continue

        image_array = read_image(row["path"])
        plt.imshow(image_array)
        plt.axis("off")
        plt.show(block=False)

        while True:
            # Collect 0/1 answers for this image
            input_map = {"fname": fname}

            for col in manual_columns:
                while True:
                    ans = input(f"0/1 | {col}: ").strip()
                    if ans in ["0", "1"]:
                        input_map[col] = int(ans)
                        break
                    print("Invalid input. Please enter 0 or 1.")

            # Show summary and ask for confirmation
            print("\nYou entered:")
            for col in manual_columns:
                print(f"  {col}: {input_map[col]}")

            while True:
                confirm = input("Confirm this annotation? (y = save, r = redo): ").strip().lower()
                if confirm in ["y", "r"]:
                    break
                print("Invalid input. Please enter 'y' to save or 'r' to redo.")

            if confirm == "y":
                # Accept and save this image's labels
                break
            else:
                # Redo this image's labels
                print("Okay, let's redo this image.\n")

        plt.close()

        # Append and save immediately so progress isn't lost
        tagged_df = pd.concat(
            [tagged_df, pd.DataFrame([input_map])],
            ignore_index=True
        )
        tagged_df.to_csv(csv_path, index=False)
        tagged_files.add(fname)

        print(f"Saved progress. {len(tagged_df)}/{len(df)} complete.")


if __name__ == "__main__":
    tag_images()
