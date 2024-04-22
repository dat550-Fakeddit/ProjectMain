import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Multi_Modal_Dataset(Dataset):

    def __init__(self, img_dir, text_tokenizer, image_transform, classes=2, df_labels=None):

        # make sure directory exists
        if not os.path.exists(img_dir):
            raise ValueError(f"{img_dir} does not exist")

        # Argument validations
        if df_labels is None:
            raise ValueError("dataframe with labels must be provided")
        if classes not in [2, 3, 6]:
            raise ValueError("classes must be 2, 3, or 6")

        # transform to apply to images
        self.transform = image_transform

        # max sequence length
        self.max_seq_len = text_tokenizer.max_padding_size

        def padded_tokenizer(text):
            return text_tokenizer.encode(
                text, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=text_tokenizer.max_padding_size
            ).squeeze()
        
        self.tokenizer = padded_tokenizer
        

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
    
    def get_class_label(self, image_path):
        id = os.path.basename(image_path).strip().split(".")[0]
        label = self.df_labels.loc[id, self.n_way_label]
        return torch.tensor(label, dtype=torch.long)

    def get_tokens(self, image_path):
        id = os.path.basename(image_path).strip().split(".")[0]
        text = self.df_labels.loc[id, "clean_title"]
        return self.tokenizer(text)

    def get_images(self, image_path):
        x = Image.open(image_path).convert("RGB")
        return self.transform(x)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        images = self.get_images(image_path)
        tokens = self.get_tokens(image_path)
        labels = self.get_class_label(image_path)
        return tokens, images, labels
 
    def __len__(self):
        return len(self.image_paths)


