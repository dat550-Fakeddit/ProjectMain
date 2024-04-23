# Data set generator for fakeddit data

The data set generator is made for generating balanced datasets from the fakeddit data.
Fakeddit: https://github.com/entitize/Fakeddit

Follow the instructions below to set up your environment for generating datasets.

Make sure you have installed all requirements as stated in the main [README.md](../README.md)

### Download dataset

You will need to download the entire image set. Follow **OPTION 1** to download all images [here](https://github.com/entitize/Fakeddit?tab=readme-ov-file#download-image-data).

Fakeddit provides a link to download a "v2.0 dataset" from google drive [here](https://github.com/entitize/Fakeddit?tab=readme-ov-file#download-text-and-metadata). In the google drive, there are two options. You must download the "multimodal_only_samples".

Extract the images and the text dataset, and place them into the full_set folder. The structure should look like:

```
project
|
└───data
|   |   README.md
|   |   data_set_generator.py
|   |
|   └───full_set
|   |   |   multimodal_train.tsv
|   |   |   multimodal_test_public.tsv
|   |   |   multimodal_validate.tsv
|   |   |
|   |   └───public_image_set
|   |       |   <imageId>.jpg
|   |       |   <imageId2>.jpg
|   |       |   ...
|   |
|   └───<Generated-set>
|   |
|   └───<Generated-set2>
|
└───...
```

### Generate datasets

In the [data_set_generator.py](data_set_generator.py) file, you can customize:

- The count
    - How large should the dataset be?
- The category
    - Which category. It can be either "2_way_label", "3_way_label" and "6_way_label".
- The counters
    - These makes sure that each label has the same amount of entries.
    - The length of this array should be equal to the number of labels in that category.
    - e.g
    ```py
    category = "2_way_label" 
    counters = [0, 0]
    ```

If the count (the requested number of entries in the new dataset) is for example 60 000, then 30 000 of each label (provided 2_way_label category) will be present in the new train dataset.

The script also automatically generates datasets for test and validate. The size of these datasets will be 20 % the size of the train dataset.

**NB**: There are cases where there may not be enough labels in a given category to satisfy the requested size. In this case, the dataset will still be generated, however, there will be missing entries for that label and thus be smaller than the requested size. This also causes the dataset to not be balanced.