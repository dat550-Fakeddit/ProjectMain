# Multimodal Fakeddit Project dat550
## Getting started
This project are dependent on some python packages. Make sure they are installed. You can use `pip install -r ./requirements.txt`

### Dataset
Follow the instructions provided in the [README.md](data/README.md)

## Training a model - Non clip
### Configuration
The config file, [modules/config.py](modules/config.py), contains the setup for several vision and text encoder models, including tokenizer and image transformation methods, as well as other configuration options.

At line 117, there are configuration options which must be set.
- Dataset
    - When generating a dataset, it makes a folder with that dataset. E.g `60000_2_way_label`.
    - Set dataset to the name of this folder. E.g `DATASET = "60000_2_way_label"`
- Classes
    - What classification-category to use. E.g For 6-way-labels, use 6 classes
- Batch size
- Epochs
- Vison_model
    - What vision model to use. E.g to use ViT, set `VISION_MODEL = ViT_model`
- Text_encoder_model
    - What text encoder model to use. E.g to use bert base uncased, set `TEXT_ENCODER_MODEL = bert_base_uncased_model`
- Tokenizer
    - What tokenizer to use. It should be set to the tokenizer made for the chosen text encoder. E.g `TOKENIZER = bert_base_uncased_tokenizer`
- Combine_method
    - What combine function to use on the outputs of the models.
    - Concatinate, Average, Maximum or Add
- Wandb configuration
    - Provide a project name, model name and api key for Wandb logging

### Start training session
Run the [Multimodal_TRAIN.py](Multimodal_TRAIN.py) script to start a training session. There are however some parameters in this file which may be tweaked.

Within each Dataloader definition, E.g at line 90, depending on your system, you may have to tweak the number of workers, persistent workers and/or pinning memory.

At line 145, you can tweak parameters such as the learning rate.

## Training a model - Clip
### Starting a Clip training session
The configuration for clip is the same as for non-clip, but is dependent on fewer parameters.

The parameters needed for Clip are:
- Classes
- Batch size
- Epochs
- Dataset
- Wandb parameters

In [Train_CLIP.py](Train_CLIP.py), you must specify a model name, which is used for saving the model, as well as for wandb logging.

Within each Dataloader definition, E.g at line 85, depending on your system, you may have to tweak the number of workers, persistent workers and/or pinning memory.

At line 125, you can tweak parameters such as the learning rate.