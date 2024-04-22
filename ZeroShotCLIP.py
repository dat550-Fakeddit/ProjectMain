import os
import PIL
import torch
import random
import pandas as pd
import numpy as np
import pandas as pd
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modules.collate_fn import collate_X_Y_Z
from modules.clip_dataset import Clip_Dataset
from transformers import CLIPProcessor, CLIPModel



# Reproducibility
seed = 1
random.seed(seed)
torch.manual_seed(seed)
L.seed_everything(seed=seed, workers=True)
torch.set_float32_matmul_precision('medium')
PIL.Image.MAX_IMAGE_PIXELS = 2809600000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the dataframe
data_dir = "data60k" #"data60k"
train_dir = os.path.join(data_dir, "train_images")            # where the images are stored to be trained on
batch_dir = os.path.join(data_dir, "batch")                 # optional to train on a subset of images
test_dir = os.path.join(data_dir, "test_images")              # validate training against test folder to find accuracy
validate_dir = os.path.join(data_dir, "validate_images")      # validate training against validate folder to find accuracy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"  # You can choose other CLIP models
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

######## TRAIN DATASET (FROM BATCH) ##############
df_train = pd.read_csv(os.path.join(data_dir, 'multimodal_train.tsv'), sep='\t')
df_train_labels = df_train[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
df_train_labels.set_index('id', inplace=True)
img_ids = [img.split('.')[0] for img in os.listdir(train_dir)] 
df_train_labels = df_train_labels.loc[img_ids]

# Create training dataset
dataset = Clip_Dataset(
    img_dir=train_dir,
    df_labels=df_train_labels,
    classes=2,
)

######## VALIDATE DATASET ########
# df_validate = pd.read_csv(os.path.join(data_dir, 'multimodal_validate.tsv'), sep='\t')
# df_validate_labels = df_validate[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
# df_validate_labels.set_index('id', inplace=True)
# validate_img_ids = [img.split('.')[0] for img in os.listdir(validate_dir)]
# df_validate_labels = df_validate_labels.loc[validate_img_ids]

# # Create training dataset
# dataset = Clip_Dataset(
#     img_dir=validate_dir,
#     df_labels=df_validate_labels,
#     classes=2,
# )

# Create the dataloader for the training dataset
dataloader = DataLoader(
    dataset,
    batch_size = 50, 
    shuffle=True,
    collate_fn=collate_X_Y_Z,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True
)



# thresholds = 0.25252525252525254 #0.25455455455455456
# correct = 0
# count = 0
# dataiter = iter(dataloader)
# for i, (X, Y, Z) in enumerate(dataloader):
#     X, Y, Z = X.to(device), Y.to(device), Z.to(device)
#     with torch.no_grad():
#         image_features = model.get_image_features(Y)
#         text_features = model.get_text_features(X)
#         cos_sim = F.cosine_similarity(image_features, text_features)
#         predicted = (cos_sim > thresholds)
#         correct += (predicted == Z).sum().item()
#         count += predicted.size(0)
#         print(f"Progress: batches = {count}/{len(dataset)} thresshold: {thresholds}  Accuracy: {correct/count:.5f}")


thresholds = {key: [0,0] for key in np.linspace(0, 1, num=100)}
dataiter = iter(dataloader)
for i, (X, Y, Z) in enumerate(dataloader):
    X, Y, Z = X.to(device), Y.to(device), Z.to(device)
    with torch.no_grad():
        image_features = model.get_image_features(Y)
        text_features = model.get_text_features(X)
        cos_sim = F.cosine_similarity(image_features, text_features)
        for key in thresholds.keys():
            predicted = (cos_sim > key)
            thresholds[key][0] += (predicted == Z).sum().item()
            thresholds[key][1] += predicted.size(0)
        highest_key, (highest_value, count) = max(thresholds.items(), key=lambda x: x[1][0])
        print(f"Progress: batches = {count}/{len(dataset)} thresshold: {highest_key}  Accuracy: {highest_value/count:.2f}")

highest_key, (highest_value, count) = max(thresholds.items(), key=lambda x: x[1][0])
print(f"Progress: batches = {count}/{len(dataset)} thresshold: {highest_key}  Accuracy: {highest_value/count:.2f}")

print(thresholds)
acc_dict = {key: value[0]/value[1] for key, value in thresholds.items()}
df = pd.DataFrame.from_dict(acc_dict, orient='index', columns=['Accuracy'])
df = df.rename_axis('Threshold')  # Rename the index

#df.to_csv('CSV_FILES/2-way-ZeroShotCLIP.csv')