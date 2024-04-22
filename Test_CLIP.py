import os
import torch
import numpy as np
import random
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from modules.collate_fn import collate_X_Y_Z
from modules.multimodal_classifier import CLIPModel_Classifier
from modules.clip_dataset import Clip_Dataset

# Reproducibility
seed = 1
L.seed_everything(seed=seed, workers=True)
torch.manual_seed(seed)
random.seed(seed)


data_dir = "data"
train_dir = os.path.join(data_dir, "train_images")            # where the images are stored to be trained on
batch_dir = os.path.join(data_dir, "batch")                 # optional to train on a subset of images
test_dir = os.path.join(data_dir, "test_images")              # validate training against test folder to find accuracy
validate_dir = os.path.join(data_dir, "validate_images")      # validate training against validate folder to find accuracy


# Set the device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cur_dir = test_dir
cur_data = "data/multimodal_test_public.tsv"
df_test = pd.read_csv(cur_data, sep='\t')
df_test_labels = df_test[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
df_test_labels.set_index('id', inplace=True)
test_img_ids = [img.split('.')[0] for img in os.listdir(cur_dir)]
df_test_labels = df_test_labels.loc[test_img_ids]


MODEL_NAME = "trained_models_6_way/6-way-CLIP-FREEZE-concat-lr=0.2-wd=1e-05-bs=50.pt"

# Load checkpoint
checkpoint = torch.load(MODEL_NAME)
model_state_dict = checkpoint['model_state_dict']
model_args = checkpoint['model_args']
dataset_args = checkpoint['dataset_args']

# Generate model
loaded_model = CLIPModel_Classifier(**model_args).to(device)

# load weights
loaded_model.load_state_dict(model_state_dict)
loaded_model.eval()



classes = 2
correct = 0
predictions = {
    0: 0,
    1: 0
}

cur_dataset = Clip_Dataset(
    img_dir=cur_dir,
    df_labels=df_test_labels,
    **dataset_args
)


dataloader = DataLoader(
    cur_dataset,
    batch_size=50,
    shuffle=True,
    collate_fn=collate_X_Y_Z,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True
)

dataiter = iter(dataloader)
correct = 0
count = 0
for i, (X, Y, Z) in enumerate(dataloader):
    X, Y, Z = X.to(device), Y.to(device), Z.to(device)
    with torch.no_grad():
        out = loaded_model(X, Y)
        predicted = torch.argmax(out, dim=1)
        correct += (predicted == Z).sum().item()
        count += len(Z)
        print(f'Progress: {count}/{len(cur_dataset)} | Test Accuracy: {np.round(correct/count, 3)}')