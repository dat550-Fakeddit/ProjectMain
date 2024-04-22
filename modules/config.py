import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel 
from transformers import AlbertTokenizer, AlbertModel
from transformers import ViTImageProcessor, ViTModel
from transformers import CLIPModel, CLIPProcessor



####### Vision models ########


# Efficientnetv2
load_weights_name = 'EfficientNet_V2_S_Weights.DEFAULT'
efficientnet_v2_s = models.efficientnet_v2_s(weights=load_weights_name)
efficientnet_v2_s.load_weights_name = load_weights_name.replace("/", "-")
efficientnet_v2_s.image_out_size  = efficientnet_v2_s.classifier[-1].in_features
efficientnet_v2_s.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Identity())
efficientnet_v2_s.image_transform = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR), 
    transforms.CenterCrop(384),                                                
    transforms.ToTensor(),                                                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])



# Resnet50
load_weights_name = 'ResNet50_Weights.DEFAULT'
resnet50 = models.resnet50(weights=load_weights_name)
resnet50.load_weights_name = load_weights_name.replace("/", "-")
resnet50.image_out_size = resnet50.fc.in_features
resnet50.fc = nn.Identity()
resnet50.image_transform = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ViT
load_weights_name = "google/vit-base-patch16-224-in21k"
ViT_model = ViTModel.from_pretrained(load_weights_name)
ViT_model.load_weights_name = load_weights_name.replace("/", "-")
ViT_processor = ViTImageProcessor.from_pretrained(load_weights_name)
ViT_model.image_out_size = ViT_model.config.hidden_size

# Extract features from output
def extract_features(x):
    return torch.mean(x.last_hidden_state, dim=1)
ViT_model.extract_features = extract_features

# Process image
def image_transform(image):
    image = transforms.Resize((ViT_processor.size["height"], ViT_processor.size["width"]))(image)
    return ViT_processor(image, return_tensors="pt").pixel_values.squeeze()
ViT_model.image_transform = image_transform

# Freeze ViT parameters
# for param in ViT_model.parameters():
#     param.requires_grad = False


####### Text Encoders ########

# Bert
load_weights_name = "bert-base-cased"
bert_base_cased_model = BertModel.from_pretrained(load_weights_name)
bert_base_cased_model.load_weights_name = load_weights_name
bert_base_cased_model.text_out_size = bert_base_cased_model.config.hidden_size
bert_base_cased_tokenizer = BertTokenizer.from_pretrained(load_weights_name)
bert_base_cased_tokenizer.max_padding_size = 256


load_weights_name = "bert-base-uncased"
bert_base_uncased_model = BertModel.from_pretrained(load_weights_name)
bert_base_uncased_model.load_weights_name = load_weights_name
bert_base_uncased_model.text_out_size = bert_base_uncased_model.config.hidden_size
bert_base_uncased_tokenizer = BertTokenizer.from_pretrained(load_weights_name)
bert_base_uncased_tokenizer.max_padding_size = 256

load_weights_name = "bert-large-uncased"
bert_large_uncased_model = BertModel.from_pretrained(load_weights_name)
bert_large_uncased_model.load_weights_name = load_weights_name
bert_large_uncased_model.text_out_size = bert_large_uncased_model.config.hidden_size
bert_large_uncased_tokenizer = BertTokenizer.from_pretrained(load_weights_name)
bert_large_uncased_tokenizer.max_padding_size = 256


# Roberta
load_weights_name = "FacebookAI/roberta-base"
roberta_model = RobertaModel.from_pretrained(load_weights_name,add_pooling_layer=False)
roberta_model.load_weights_name = load_weights_name
roberta_model.text_out_size = roberta_model.config.hidden_size
roberta_tokenizer = RobertaTokenizer.from_pretrained(load_weights_name)
roberta_tokenizer.max_padding_size = 256

# Albert
load_weights_name = "albert-base-v2"
albert_model = AlbertModel.from_pretrained(load_weights_name)
albert_model.load_weights_name = load_weights_name
albert_model.text_out_size = albert_model.config.hidden_size
albert_tokenizer = AlbertTokenizer.from_pretrained(load_weights_name)
albert_tokenizer.max_padding_size = 256


# Parse model name
def get_model_name(text_model, vision_model, classes, combine_method):
    text_name = text_model.load_weights_name.replace("/", "-")
    vision_name = vision_model.load_weights_name.replace("/", "-")
    return f"{classes}-way-{text_name}-{vision_name}-{combine_method}"


# Configurations for training
CLASSES = 2
BATCH_SIZE = 50
EPOCHS = 30 # 30


# TEXT_ENCODER_MODEL = bert_base_uncased_model #roberta_model, albert_model, bert_base_cased_model, bert_large_uncased_model
# VISION_MODEL = efficientnet_v2_s
# TOKENIZER = bert_base_uncased_tokenizer # roberta_tokenizer, albert_tokenizer, bert_base_cased_tokenizer, bert_large_uncased_tokenizer

VISION_MODEL = efficientnet_v2_s
TEXT_ENCODER_MODEL = bert_base_uncased_model
TOKENIZER = bert_base_uncased_tokenizer


def combine_func(text_x, img_x, method):
    if method == "concat":
        return torch.cat((img_x, text_x), dim=1)
    elif method == "avg":
        return torch.mean(torch.stack([img_x, text_x]),dim=0)
    elif method == "max":
        return torch.max(img_x, text_x)
    elif method == "add":
        return img_x + text_x

COMBINE_INFO = {
    "combine_func": combine_func,
    "method": "concat"
}

#WANDB_PROJECT = "dat550_Text_Encoder"
WANDB_PROJECT = "TESTING"#"dat550_2_Way_Vision_Model" # dat550_Vision_Model
WANDB_MODEL_NAME = f"{get_model_name(TEXT_ENCODER_MODEL, VISION_MODEL, CLASSES, COMBINE_INFO['method'])}"
MODEL_NAME = f"trained_models/{WANDB_MODEL_NAME}.pt"
WANDB_API_KEY = "57bc17c6e9359ec344d6b67283f22f6c6dcc09eb"

# Configurations for testing
#model_name = get_model_name(TEXT_ENCODER_MODEL, VISION_MODEL, CLASSES)
model_name = "2-way-bert-base-uncased-EfficientNet_V2_S_Weights.DEFAULT-concat"
#model_name = "2-way-bert-base-cased-EfficientNet_V2_S_Weights.DEFAULT-concat"
TEST_MODEL_NAME = f"trained_models/{model_name}.pt"


if __name__ == "__main__":
    pass