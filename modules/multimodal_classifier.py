import torch
import wandb
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPModel, CLIPConfig



class Multimodal_Classifier(L.LightningModule):
    
    def __init__(
        self, 
        input_size, 
        num_channels, 
        classes, 
        text_tokenizer, 
        text_encoder_model, 
        vision_model, 
        configure_parameters,
        combine_info
    ):

        super().__init__()

        ######### Parameters #########
        self.configure_parameters = configure_parameters
        self.input_size = input_size
        self.num_channels = num_channels
        self.classes = classes

        ######### Criterion #########
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()


        ########## Text Encoder #########
        self.text_tokenizer = text_tokenizer
        self.text_encoder_model = text_encoder_model
        text_out_size = self.text_encoder_model.text_out_size
    

        ######### Vision Model #########
        self.vision_model = vision_model
        image_out_size = self.vision_model.image_out_size
     

        ######### Dropout #########
        self.dropout = nn.Dropout(p=0.2)


        ######## Adjust dimensions ##########
        self.combine_info = combine_info
        self.adjust_dim = None
        self.fc_out_size =  image_out_size + text_out_size
        if self.combine_info["method"] != "concat":
            max_size = max(image_out_size, text_out_size)
            min_size = min(image_out_size, text_out_size)
            self.adjust_dim = nn.Linear(min_size, max_size)
            self.fc_out_size = max_size

        ######### Fully connected layers #########
        self.fc = nn.Linear(self.fc_out_size, classes)

        print("\n\n############################ MODEL INFO ###################################")
        print("image_out_size: ",image_out_size)
        print("text_out_size: ",text_out_size)
        print("self.adjust_dim: ", self.adjust_dim)
        print("self.fc: ",self.fc)
        print("###############################################################################\n\n")


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.configure_parameters["learning_rate"], 
            weight_decay=self.configure_parameters["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.configure_parameters["max_learning_rate"], 
            steps_per_epoch=self.configure_parameters["steps_per_epoch"], 
            epochs=self.configure_parameters["max_epochs"]
        )
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [self.optimizer], [self.scheduler]


    def train_dataloader(self):
        return self.dataloader 


    def calculate_metrics(self, predictions, labels):
        # Calculate accuracy
        predicted = torch.argmax(predictions, dim=1)
        correct = torch.sum(predicted == labels).item()
        accuracy = correct / labels.size(0)

        # Calculate precision, recall, and f1
        true_positives = torch.sum((predicted == labels) & (predicted == 1)).item()
        false_positives = torch.sum((predicted == 1) & (labels == 0)).item()
        false_negatives = torch.sum((predicted == 0) & (labels == 1)).item()

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return accuracy, precision, recall, f1


    def training_step(self, batch, batch_idx):
        tokens, images, labels = batch
       
        predictions = self(tokens, images)
        loss = self.criterion(predictions, labels)
        accuracy, precision, recall, f1 = self.calculate_metrics(predictions, labels)

        # Log metrics
        self.log("train-accuracy", accuracy, on_epoch=True)#, prog_bar=True)
        self.log("train-precision", precision, on_epoch=True)#, prog_bar=True)
        self.log("train-recall", recall, on_epoch=True)#, prog_bar=True)
        self.log("train-f1", f1, on_epoch=True)#, prog_bar=True)
        self.log("train-loss", loss, on_epoch=True)#, prog_bar=True)

        return loss


    def on_train_epoch_end(self):
        
        self.eval()

        test_dataloader = self.test_dataloader
        loss, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for i, (X, Y, Z) in enumerate(test_dataloader):
                X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
                predicted = self(X, Y)
                loss += self.criterion(predicted, Z)
                accuracy_batch, precision_batch, recall_batch, f1_batch = self.calculate_metrics(predicted, Z)
                accuracy += accuracy_batch
                precision += precision_batch
                recall += recall_batch
                f1 += f1_batch
        
        accuracy /= len(test_dataloader)
        loss /= len(test_dataloader)
        precision /= len(test_dataloader)
        recall /= len(test_dataloader)
        f1 /= len(test_dataloader)

        self.log("test-accuracy", accuracy, on_epoch=True)
        self.log("test-precision", precision, on_epoch=True)
        self.log("test-recall", recall, on_epoch=True)
        self.log("test-f1", f1, on_epoch=True)
        self.log("test-loss", loss, on_epoch=True)
                    
        
        validate_dataloader = self.validate_dataloader
        loss, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for i, (X, Y, Z) in enumerate(validate_dataloader):
                X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
                predicted = self(X, Y)
                loss += self.criterion(predicted, Z)
                accuracy_batch, precision_batch, recall_batch, f1_batch = self.calculate_metrics(predicted, Z)
                accuracy += accuracy_batch
                precision += precision_batch
                recall += recall_batch
                f1 += f1_batch

        accuracy /= len(validate_dataloader)
        loss /= len(validate_dataloader)
        precision /= len(validate_dataloader)
        recall /= len(validate_dataloader)
        f1 /= len(validate_dataloader)

        self.log("validate-accuracy", accuracy, on_epoch=True)
        self.log("validate-precision", precision, on_epoch=True)
        self.log("validate-recall", recall, on_epoch=True)
        self.log("validate-f1", f1, on_epoch=True)
        self.log("validate-loss", loss, on_epoch=True)

        self.train()

    
    def forward(self, tokens, image):

        ####### Forward pass through vision model #####
        image_x = self.vision_model(image)
        if hasattr(self.vision_model, 'extract_features'):
            image_x = self.vision_model.extract_features(image_x)
        image_x = image_x.view(image_x.size(0), -1)
        #####################################
        
        ####### Forward pass through BERT #####
        attention_mask = (tokens != self.text_tokenizer.pad_token_id).to(torch.long)
        
        # Take average of all embedded layers
        text_x = torch.mean(self.text_encoder_model(tokens, attention_mask=attention_mask).last_hidden_state, dim=1)
        #text_x = self.text_encoder_model(tokens, attention_mask=attention_mask).last_hidden_state[:,-1,:]
        #####################################
        

        ###### Adjust dimensions and add ReLU activation #####
        if self.combine_info["method"] != "concat":
            if text_x.size(1) < image_x.size(1):
                text_x = self.adjust_dim(text_x)
                image_x = image_x
            elif image_x.size(1) > text_x.size(1):
                image_x = self.adjust_dim(image_x)
                text_x = text_x
        ######################################################


        # Concatenate the outputs of the two models
        concat_x = self.combine_info["combine_func"](text_x, image_x, self.combine_info["method"])
        
        # Add dropout layer
        concat_x = self.dropout(concat_x)
        
        # Fully connected layer
        out = self.fc(concat_x)
        
        return out



# https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/clip/modeling_clip.html
class CLIPModel_Classifier(L.LightningModule):
    def __init__(self, num_channels, classes, configure_parameters):
        super().__init__()

        ######### Parameters #########
        self.configure_parameters = configure_parameters

        ######### Parameters #########
        self.num_channels = num_channels
        self.classes = classes

        ######### Criterion #########
        self.criterion = nn.CrossEntropyLoss()

        ########## CLIP #########
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # config = CLIPConfig(
        #     text_config=model.text_model.config.to_dict(),
        #     vision_config=model.vision_model.config.to_dict(),
        #     projection_dim=512,
        #     logit_scale_init_value=2.6592,
        # )
        # self.clip_model = CLIPModel(config)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        image_out_size =  self.clip_model.visual_projection.out_features
        text_out_size =  self.clip_model.text_projection.out_features

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    
        ######### Dropout #########
        self.dropout = nn.Dropout(p=0.2)

        ######### Fully connected layers #########
        self.fc = nn.Linear(image_out_size+text_out_size, classes)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.configure_parameters["learning_rate"], 
            weight_decay=self.configure_parameters["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.configure_parameters["max_learning_rate"], 
            steps_per_epoch=self.configure_parameters["steps_per_epoch"], 
            epochs=self.configure_parameters["max_epochs"]
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataloader 

    def on_train_epoch_end(self):

        self.eval()

        test_dataloader = self.test_dataloader
        test_correct, test_count, test_loss = 0, 0, 0
        with torch.no_grad():
            for i, (X, Y, Z) in enumerate(test_dataloader):
                X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
                out = self(X, Y)
                test_loss += self.criterion(out, Z)
                predicted = torch.argmax(out, dim=1)
                test_correct += (predicted == Z).sum().item()
                test_count += Z.size(0)
                
        
        validate_dataloader = self.validate_dataloader
        validate_correct, validate_count, validate_loss = 0, 0, 0
        with torch.no_grad():
            for i, (X, Y, Z) in enumerate(validate_dataloader):
                X, Y, Z = X.to(self.device), Y.to(self.device), Z.to(self.device)
                out = self(X, Y)
                validate_loss += self.criterion(out, Z)
                predicted = torch.argmax(out, dim=1)
                validate_correct += (predicted == Z).sum().item()
                validate_count += Z.size(0)

        self.log("test-loss", test_loss/len(test_dataloader))
        self.log("test-accuracy: ",test_correct/test_count)
        
        self.log("validate-loss", validate_loss/len(validate_dataloader))
        self.log("validate-accuracy: ",validate_correct/validate_count)

        self.train()

    def calculate_metrics(self, predictions, labels):
        # Calculate accuracy
        predicted = torch.argmax(predictions, dim=1)
        correct = torch.sum(predicted == labels).item()
        accuracy = correct / labels.size(0)

        # Calculate precision, recall, and f1
        true_positives = torch.sum((predicted == labels) & (predicted == 1)).item()
        false_positives = torch.sum((predicted == 1) & (labels == 0)).item()
        false_negatives = torch.sum((predicted == 0) & (labels == 1)).item()

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return accuracy, precision, recall, f1

    def training_step(self, batch, batch_idx):
        tokens, images, labels = batch
       
        predictions = self(tokens, images)
        loss = self.criterion(predictions, labels)
        accuracy, precision, recall, f1 = self.calculate_metrics(predictions, labels)

        # Log metrics
        self.log("train-accuracy", accuracy, on_epoch=True)#, prog_bar=True)
        self.log("train-precision", precision, on_epoch=True)#, prog_bar=True)
        self.log("train-recall", recall, on_epoch=True)#, prog_bar=True)
        self.log("train-f1", f1, on_epoch=True)#, prog_bar=True)
        self.log("train-loss", loss, on_epoch=True)#, prog_bar=True)

        return loss
    
    def forward(self, tokens, images):

        # Unnormalized embeds    
        image_features = self.clip_model.get_image_features(images)
        text_features = self.clip_model.get_text_features(tokens)

        # Normalized embeds
        outputs = self.clip_model(pixel_values=images, input_ids=tokens)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # concatenate embeddings   
        concat_x = torch.concat([image_features,text_features],dim=1)

        # Add dropout layer
        concat_x = self.dropout(concat_x)

        # Fully connected layer
        out = self.fc(concat_x)
        
        return out

    
if __name__ == "__main__":
    pass
    