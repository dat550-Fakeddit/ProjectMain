# slurm_train Old

- sbatch conda_setup.sh
- sbatch update_pip.sh

- sbatch Multimodal_TRAIN.sh
- sbatch Multimodal_TEST.sh

- sbatch Train_CLIP.sh
- sbatch Test_CLIP.sh

# Multimodal Fakeddit project

Names
References
General information

## Getting started

Requirements
File structure

## Dataset

### Download dataset

Download the images and dataset as specified in https://github.com/entitize/Fakeddit

Use **OPTION 1** to download all images https://github.com/entitize/Fakeddit?tab=readme-ov-file#download-image-data

Download the Multimodal only datasets https://github.com/entitize/Fakeddit?tab=readme-ov-file#download-text-and-metadata

In the data folder, add the datasets into the "original_data" folder. Add the images to "original_data/images".

### Generate dataset

Run the "generate_dataset.py" to generate the dataset.

## Train model

Explain

### Config

### Parameters
