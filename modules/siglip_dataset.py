import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import SiglipProcessor




class SigLIP_Dataset(Dataset):

    def __init__(self, img_dir, classes=2, df_labels=None):

        # make sure directory exists
        if not os.path.exists(img_dir):
            raise ValueError(f"{img_dir} does not exist")

        # Argument validations
        if df_labels is None:
            raise ValueError("dataframe with labels must be provided")
        if classes not in [2, 3, 6]:
            raise ValueError("classes must be 2, 3, or 6")

        self.processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-256-i18n")

        # get images in a directory
        self.image_paths = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]


        # dataframe to fetch labels
        self.df_labels = df_labels

        # keep track of distribution of classes
        self.n_way_label = "2_way_label"
        if classes == 3:
            self.n_way_label = "3_way_label"
        elif classes == 6:
            self.n_way_label = "6_way_label"
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_id = os.path.basename(image_path).strip().split(".")[0]
        text_data = self.df_labels.loc[sample_id, "clean_title"]
        image_data = Image.open(image_path).convert("RGB")
        label = self.df_labels.loc[sample_id, self.n_way_label]
        
        tokens = self.processor(text=text_data, return_tensors="pt", padding="max_length", max_length=64, truncation=True).input_ids
        images = self.processor(images=image_data, return_tensors="pt").pixel_values
        labels = torch.tensor(self.df_labels.loc[sample_id, self.n_way_label], dtype=torch.long)
        return tokens.squeeze(), images.squeeze(), labels
 
    def __len__(self):
        return len(self.image_paths)

    
if __name__ == "__main__":
    pass