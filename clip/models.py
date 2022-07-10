import torch 
import torchvision.models as models
from torch import nn


import timm
from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, pipeline

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.backbone = timm.create_model("resnet18", num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("klue/roberta-small", num_labels=512)
        self.model = self.model.eval()

        self.projection = torch.nn.Linear(512, 512)
        self.projection = self.projection.train()

    def forward(self, input_ids, attention_mask):
        
        x = self.model(input_ids, attention_mask=attention_mask)[0]
        x = self.projection(x)
        return x