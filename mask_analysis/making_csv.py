import os
import pandas as pd
from natsort import natsorted

# Creates a csv file with the name of all the pictures with a column to manually check if the image should be removed
# and a column checking if a mask needs to be made for that image either because there is none or the existing is insufficient

base_path = os.path.dirname(__file__)

folder_path = os.path.join(base_path, "..", "data", "skin_images")

save_path = base_path
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
file_names = natsorted(file_names)

df = pd.DataFrame(file_names, columns=["Filename"])

df["Exclude"] = 0
df["Remake mask"] = 0

output_path = os.path.join(save_path, "mask_analysis.csv")
df.to_csv(output_path, index=False)
