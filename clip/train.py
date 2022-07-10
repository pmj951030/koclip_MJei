import torch
import clip
import random
import argparse
from torch.autograd.grad_mode import no_grad

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision

import math
import time
import numpy as np
# import wandb
import clip
from transformers import AutoModel, AutoTokenizer

from models import ImageEncoder, TextEncoder
from datasets import ImageTextPairDataset
parser = argparse.ArgumentParser(description="Korean Image Text Clip Implementation")

parser.add_argument("--epochs", default=100, type=int,
                help="epochs of training")
parser.add_argument("--batch_size", default=192, type=int,
                help="batch size of training")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')       


args = parser.parse_args()



if __name__ == "__main__":
    random.seed(42)
    
#     wandb.init(project="ko-clip", entity="easter3163")


    train_dataset = ImageTextPairDataset() # define in dataset.py

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    # define in models.py 
    # image encoder includes Projection Head, So dimension size is 512
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Validation : Cifar10 
    validation_dataset = torchvision.datasets.CIFAR10(root='D:/coco_data/data', train=False,
                                       download=True, transform=preprocess)
    testloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    text_encoder = TextEncoder().to(device)
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)


    optimizer = optim.SGD(text_encoder.parameters(), lr=args.lr,
               momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")

    ce = torch.nn.CrossEntropyLoss()
    text_encoder.train()

    max_acc = 0

    for epoch in range(args.epochs):
        start = time.time()
        loss_for_monitoring = 0

        for idx, (batch_img, batch_input_ids, batch_attention_mask) in enumerate(trainloader):
            
            with no_grad():
                image_embedding = clip_model.encode_image(batch_img.cuda()).float() # Output : N x 512
            
            text_embedding = text_encoder(batch_input_ids.cuda(), batch_attention_mask.cuda()).float() # Output : N x 512

            
            # Normalization is need for calculating cosine similarity
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)    
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            # optimizer
            optimizer.zero_grad()

            loss = 0

            image_to_text = (image_embedding @ text_embedding.T) * math.exp(0.07) # (N x 512) x (512 x N) = N x N
            text_to_image = (text_embedding @ image_embedding.T) * math.exp(0.07) # (N x 512) x (512 x N) = N x N, 0.07 means temperature

            # Optional : How to add the self-supervised loss?

            # Temperature Normalized Cross Entropy loss
            label = torch.arange(args.batch_size, dtype=torch.long).cuda() # Label is 0~N-1 Why? : Because Batch pair (i, i) is Positive, Other pair (i, j) is Negative


            loss = (ce(image_to_text, label) + ce(text_to_image, label)) * 0.5
            loss.backward()

            optimizer.step()
            
            loss_for_monitoring += loss.item()
#             wandb.log({"Train Iteration Loss" : loss.item()})

            if idx % 100 == 0:
                print("Batch : {}, Image text loss : {:.5f}".format(idx, loss.item()))
                
            
        # How we determine our best model?

        scheduler.step()

        print("Epoch : {:2d} , image text loss : {:.5f} , Time : {}".format(epoch, loss_for_monitoring / len(trainloader), time.time() - start))
#         wandb.log({"Train Epoch Loss" : loss_for_monitoring / len(trainloader)})

        total = 0
        correct = 0
        
        korean_labels = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]

        batch_input_ids, batch_attention_mask = [], []
        for korean_label in korean_labels:
            text_tensor = tokenizer(
                korean_label,
                return_tensors='pt',
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                add_special_tokens=True,
                return_token_type_ids=False
            )
            unit_input_id = text_tensor['input_ids'][0]
            unit_attention_mask = text_tensor['attention_mask'][0]
            batch_input_ids.append(unit_input_id.unsqueeze(0))
            batch_attention_mask.append(unit_attention_mask.unsqueeze(0))
        
        
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)

        text_embedding = text_encoder(batch_input_ids.cuda(), batch_attention_mask.cuda()).float() 
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        for data in testloader:
            images, labels = data
            image_embedding = clip_model.encode_image(images.cuda()).float()
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            
            image_to_text = (image_embedding @ text_embedding.T) * math.exp(0.07)
            _, predictions = torch.max(image_to_text, 1)
            
            for label, prediction in zip(labels, predictions):
                if label.cuda() == prediction:
                    correct += 1
                total += 1
        print("Accuracy : {:.3f}".format(correct / total))
#         wandb.log({"Validation Acc" : correct / total})

        acc = correct / total
        if max_acc < acc:
            max_acc = acc
            torch.save(text_encoder.state_dict(), "D:/coco_data/text_encoder.pth")