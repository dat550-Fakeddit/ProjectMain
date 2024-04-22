import os
import PIL
import torch
import wandb
import random
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from modules.collate_fn import collate_X_Y_Z
from pytorch_lightning.loggers import WandbLogger
from modules.multimodal_classifier import CLIPModel_Classifier
from modules.clip_dataset import Clip_Dataset

print('Test1')

from modules.config import (
    WANDB_API_KEY, 
    BATCH_SIZE,
    EPOCHS,
    CLASSES,
)

MODEL_NAME = "6-way-CLIP-freeze-normalized"

# Reproducibility
seed = 1
random.seed(seed)
torch.manual_seed(seed)
L.seed_everything(seed=seed, workers=True)
torch.set_float32_matmul_precision('medium')
PIL.Image.MAX_IMAGE_PIXELS = 1809600000

# load the dataframe
data_dir = "6_way_dataset60k"
train_dir = os.path.join(data_dir, "train_images")            # where the images are stored to be trained on
batch_dir = os.path.join(data_dir, "batch")                 # optional to train on a subset of images
test_dir = os.path.join(data_dir, "test_images")              # validate training against test folder to find accuracy
validate_dir = os.path.join(data_dir, "validate_images")      # validate training against validate folder to find accuracy


######## TRAIN DATASET (FROM BATCH) ##############
df_train = pd.read_csv(os.path.join(data_dir, 'multimodal_train.tsv'), sep='\t')
df_train_labels = df_train[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
df_train_labels.set_index('id', inplace=True)
img_ids = [img.split('.')[0] for img in os.listdir(train_dir)]
df_train_labels = df_train_labels.loc[img_ids]


# Create training dataset
train_dataset = Clip_Dataset(
    img_dir=train_dir,
    df_labels=df_train_labels,
    classes=CLASSES,
)

# Create the dataloader for the training dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_X_Y_Z,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True
)

print('Test2')

######## TEST DATASET ########
df_test = pd.read_csv(os.path.join(data_dir, 'multimodal_test_public.tsv'), sep='\t')
df_test_labels = df_test[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
df_test_labels.set_index('id', inplace=True)
test_img_ids = [img.split('.')[0] for img in os.listdir(test_dir)]
df_test_labels = df_test_labels.loc[test_img_ids]

# Create test dataset
test_dataset = Clip_Dataset(
    img_dir=test_dir,
    df_labels=df_test_labels,
    classes=CLASSES,
)

# Create the dataloader for the test dataset
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_X_Y_Z,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True
)

print('Test3')


######## VALIDATE DATASET ########
df_validate = pd.read_csv(os.path.join(data_dir, 'multimodal_validate.tsv'), sep='\t')
df_validate_labels = df_validate[['id','clean_title', '2_way_label', '3_way_label', '6_way_label']]
df_validate_labels.set_index('id', inplace=True)
validate_img_ids = [img.split('.')[0] for img in os.listdir(validate_dir)]
df_validate_labels = df_validate_labels.loc[validate_img_ids]

# Create validate dataset
validate_dataset = Clip_Dataset(
    img_dir=validate_dir,
    df_labels=df_validate_labels,
    classes=CLASSES,
)

# Create the dataloader for the validate dataset
validate_dataloader = DataLoader(
    validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_X_Y_Z,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True
)

steps_per_batch = len(train_dataset)//train_dataloader.batch_size
print("steps_per_batch: ",steps_per_batch)
configure_parameters = {
    "learning_rate": 0.2,
    "weight_decay": 1e-4,
    "max_learning_rate": 0.2,
    "steps_per_epoch": steps_per_batch,
    "max_epochs": EPOCHS
}

print('Test4')


MODEL_NAME = f"{MODEL_NAME}-lr={configure_parameters['learning_rate']}-wd={configure_parameters['weight_decay']}-bs={BATCH_SIZE}"

print(MODEL_NAME)

# Setup weights and biases for logging
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
try:
    print("already logged on wandb...")
    wandb.init(project="dat550_6_way_60k_dataset", reinit=True, name=MODEL_NAME)
    wandb_logger = WandbLogger()
except wandb.errors.UsageError:
    print("logging on wandb...")
    wandb.login()
    wandb.init(project="dat550_6_way_60k_dataset", reinit=True, name=MODEL_NAME)
    wandb_logger = WandbLogger()




# Create the trainer
trainer = L.Trainer(
    logger=wandb_logger, 
    max_epochs=EPOCHS, 
    accelerator="auto", 
    devices="auto",
    log_every_n_steps=steps_per_batch,
    deterministic=True,
    accumulate_grad_batches=steps_per_batch
)

# Create the multimodal model
multimodal_model = CLIPModel_Classifier(
    num_channels=3,
    classes=CLASSES,
    configure_parameters = configure_parameters
)


# Assign test and validation loaders to the model
multimodal_model.test_dataloader = test_dataloader
multimodal_model.validate_dataloader = validate_dataloader



print("\n\n############################ INFORMATION ##################################")
print(f"Training Clip model")
print(f"Classification: {CLASSES} categories")
print(f"Learning rate: {multimodal_model.configure_parameters["learning_rate"]}")
print(f"Weight decay: {multimodal_model.configure_parameters["weight_decay"]}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("###############################################################################\n\n")


# set model to train mode
multimodal_model.train()
trainer.fit(multimodal_model,train_dataloaders=train_dataloader)
wandb.finish()
#############################



# Save the weights of the model
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')


MODEL_NAME = f"trained_models/{MODEL_NAME}.pt"

torch.save({
    # weights for trained model
    'model_state_dict': multimodal_model.state_dict(),
    # Tokenizer and model used to encode text
    'dataset_args': {
        "classes": CLASSES,
    },
    # parameters to generate the model
    "model_args": {
       "configure_parameters": multimodal_model.configure_parameters,
        "num_channels": multimodal_model.num_channels,
        "classes": multimodal_model.classes,
    }
}, MODEL_NAME)


# Load checkpoint
checkpoint = torch.load(MODEL_NAME)
model_state_dict = checkpoint['model_state_dict']
model_args = checkpoint['model_args']
dataset_args = checkpoint['dataset_args']

# Generate model
loaded_model = CLIPModel_Classifier(**model_args)

# load weights
loaded_model.load_state_dict(model_state_dict)
loaded_model.eval()

print("INFERENCE ON RANDOM DATA")
X, Y, Z = next(iter(train_dataloader))
print(X.size(), Y.size(), Z.size())
out = loaded_model(X, Y)

predictions = torch.argmax(out, dim=1)
print()
print(f"Actual: {Z}")
print(f"Predicted loaded_model: {predictions}")