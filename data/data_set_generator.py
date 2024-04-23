import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys
import shutil
import PIL
from PIL import Image

# SETUP:
count = 60000 # The number of entries in the new train set.

# The category to be used for the new train set.
# Options: "2_way_label", "3_way_label", "6_way_label"
category = "3_way_label" 

# The number of entries in the new train set for each category.
# The length of the list should be equal to the number of labels for the category.
counters = [0, 0, 0]

PIL.Image.MAX_IMAGE_PIXELS = 1809600000

current_path = os.path.dirname(os.path.realpath(__file__))
main_image_path = os.path.join("E:\\", "public_images/public_image_set")



train_df = pd.read_csv(
    os.path.join(current_path, "..", "large_set", "multimodal_train.tsv"), sep="\t"
)

print(train_df.head())

new_path = os.path.join(current_path, str(count), category)
train_image_path = os.path.join(current_path, "images")

if not os.path.exists(os.path.join(current_path, str(count))):
    os.makedirs(os.path.join(current_path, str(count)))

if not os.path.exists(new_path):
    os.makedirs(new_path)

new_train_df = pd.DataFrame(columns=train_df.columns)
new_image_path = os.path.join(new_path, "train_images")

if not os.path.exists(new_image_path):
    os.makedirs(new_image_path)

for index, row in train_df.iterrows():
    if row["hasImage"] == True and row["id"] != "" and row["id"] != "nan":
        image_id = row["id"]
        image_category = row[category]
        if counters[int(image_category)] >= count / len(counters):
            continue
        if sum(counters) >= count:
            break
        if sum(counters) % 1000 == 0:
            print(sum(counters), "images processed")
        try:
            x = Image.open(os.path.join(main_image_path, image_id + ".jpg")).convert(
                "RGB"
            )
            x.save(os.path.join(new_image_path, image_id + ".jpg"))
            new_train_df.loc[len(new_train_df)] = row
            counters[int(row[category])] += 1
        except:
            print("Error in image", image_id)
            continue
new_train_df.to_csv(os.path.join(new_path, "multimodal_train.tsv"), sep="\t")
print(new_train_df[category].value_counts())

print("done train")


test_count = count * 0.2
counters = [0 for counter in counters]

test_df = pd.read_csv(
    os.path.join(current_path, "..", "large_set", "multimodal_test_public.tsv"),
    sep="\t",
)

new_test_df = pd.DataFrame(columns=test_df.columns)

new_image_path = os.path.join(new_path, "test_images")

if not os.path.exists(new_image_path):
    os.makedirs(new_image_path)

for index, row in test_df.iterrows():
    if row["hasImage"] == True and row["id"] != "" and row["id"] != "nan":
        image_id = row["id"]
        image_category = row[category]
        if counters[int(image_category)] >= test_count / len(counters):
            continue
        if sum(counters) >= test_count:
            break
        if sum(counters) % 1000 == 0:
            print(sum(counters), "images processed")
        try:
            x = Image.open(os.path.join(main_image_path, image_id + ".jpg")).convert(
                "RGB"
            )
            x.save(os.path.join(new_image_path, image_id + ".jpg"))
            new_test_df.loc[len(new_test_df)] = row
            counters[int(row[category])] += 1
        except:
            print("Error in image", image_id)
            continue
new_test_df.to_csv(os.path.join(new_path, "multimodal_test_public.tsv"), sep="\t")
print(new_test_df[category].value_counts())

print("done test")


validate_count = count * 0.2
counters = [0 for counter in counters]

validate_df = pd.read_csv(
    os.path.join(current_path, "..", "large_set", "multimodal_validate.tsv"),
    sep="\t",
)

new_validate_df = pd.DataFrame(columns=validate_df.columns)

new_image_path = os.path.join(new_path, "validate_images")

if not os.path.exists(new_image_path):
    os.makedirs(new_image_path)

for index, row in validate_df.iterrows():
    if row["hasImage"] == True and row["id"] != "" and row["id"] != "nan":
        image_id = row["id"]
        image_category = row[category]
        if counters[int(image_category)] >= validate_count / len(counters):
            continue
        if sum(counters) >= validate_count:
            break
        if sum(counters) % 1000 == 0:
            print(sum(counters), "images processed")
        try:
            x = Image.open(os.path.join(main_image_path, image_id + ".jpg")).convert(
                "RGB"
            )
            x.save(os.path.join(new_image_path, image_id + ".jpg"))
            new_validate_df.loc[len(new_validate_df)] = row
            counters[int(row[category])] += 1
        except:
            print("Error in image", image_id)
            continue
new_validate_df.to_csv(os.path.join(new_path, "multimodal_validate.tsv"), sep="\t")
print(new_validate_df[category].value_counts())

print("done validate")
