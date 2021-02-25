#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import torchvision.models as models
import cv2
from os import path
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageOps import colorize
import torch
from sklearn import model_selection
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn import preprocessing
from tqdm.notebook import trange, tqdm
# torch.cuda.set_device(torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu"))
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import gc
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


# In[18]:


nih_data = pd.read_csv("data_NIH/Data_Entry_2017.csv")
train_images = []
with open("data_NIH/train_val_list.txt", 'r') as f:
    train_images = [v.strip() for v in f.readlines()]
train = nih_data[nih_data['Image Index'].isin(train_images)]
train = train


train['path'] = train['Image Index'].apply(lambda v: path.join("data_NIH", "images", v))


classes = set()
for lab in train['Finding Labels']:
    if lab == "No Finding":
        continue
    for sl in lab.split("|"):
        classes.add(sl)
classes = sorted(list(classes))
n_classes = len(classes)

df_train, df_valid = model_selection.train_test_split(train, test_size = 0.1, random_state = 42)

df_train = df_train.reset_index(drop = True)
df_valid = df_valid.reset_index(drop = True)



class MCImageDataset:
    def __init__(
        self,
        image_paths,
        classes,
        labels,
        resize,
        torch_augs=None,
        augmentations=None,
        backend="cv2",
        channel_first= True,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.classes = classes
        self.targets = [self.make_target(lab) for lab in labels]
        self.resize = resize
        self.augmentations = augmentations
        self.torch_augs = torch_augs
        self.backend = backend
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)
    
    def make_target(self, lab):
        return torch.tensor([c in lab for c in self.classes]).float()

    def __getitem__(self, item):
        targets = self.targets[item]
        
        if self.backend == "pil":
            image = Image.open(self.image_paths[item]).convert("RGB")
            if self.resize is not None:
                image = image.resize(
                    (self.resize[1], self.resize[0]), resample=Image.BILINEAR
                )
#             image = np.array(image)
            if self.augmentations is not None:
                image = np.array(image)
                augmented = self.augmentations(image=image)
                image = augmented["image"]
            if self.torch_augs is not None:
                image = self.torch_augs(image)
        
        elif self.backend == "cv2":
            image = cv2.imread(self.image_paths[item])
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[1], self.resize[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            
            if self.augmentations is not None:
                image = np.array(image)
                augmented = self.augmentations(image=image)
                image = augmented["image"]
            if self.torch_augs is not None:
                image = self.torch_augs(image)
        
        else:
            raise Exception("Backend not implemented")
        
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        
        return [
            torch.tensor(image),
            torch.tensor(targets),
        ]


def train_vgg(epochs, train_paths, train_labs, val_paths, val_labs, mname):
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    preprocess_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
            ])

    


    train_dataset = MCImageDataset(
        image_paths=train_paths,
        classes=classes,
        labels=train_labs,
        resize= None,
        torch_augs= preprocess,
        backend="pil",
        channel_first=False
    )

    valid_dataset = MCImageDataset(
        image_paths=val_paths,
        classes=classes,
        labels=val_labs,
        resize= None,
        torch_augs=preprocess_val,
        backend="pil",
        channel_first=False
    )

    model = models.vgg16_bn(pretrained=True)


    # In[30]:


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    # In[31]:


    # set_parameter_requires_grad(model, True)
    model.classifier[6] = nn.Linear(4096,n_classes)


    # In[32]:


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=2,
        drop_last=False,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=16,
        num_workers=2,
        drop_last=False)
        
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
    criterion = nn.BCEWithLogitsLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device='cuda:0')

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device='cuda:0')

    @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def checkpoint_weights(trainer):
        torch.save(model.state_dict(), "./{}_e{}.model".format(mname, trainer.state.epoch))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        model.eval()
        tot = 0
        correct = 0
        with torch.no_grad():
            for item in train_loader:
                ims = item[0].cuda()
                labs = item[1].cuda()
                preds = model(ims)
                preds = (torch.sigmoid(preds).cpu().numpy() > 0.5).astype(float)
                right = [all(v1 == v2) for v1, v2 in zip(preds, labs.cpu().numpy())]
                tot += len(right)
                correct += sum(right)
                
        print("Epoch {} Train Accuracy: {}%".format(trainer.state.epoch, round(100 * correct / tot, 3)))
        model.train()
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        model.eval()
        tot = 0
        correct = 0
        with torch.no_grad():
            for item in valid_loader:
                ims = item[0].cuda()
                labs = item[1].cuda()
                preds = model(ims)
                preds = (torch.sigmoid(preds).cpu().numpy() > 0.5).astype(float)
                right = [all(v1 == v2) for v1, v2 in zip(preds, labs.cpu().numpy())]
                tot += len(right)
                correct += sum(right)
                
        print("Epoch {} Validation Accuracy: {}%".format(trainer.state.epoch, round(100 * correct / tot, 3)))
        model.train()
    trainer.run(train_loader, max_epochs=epochs)
    return model


for f, mname in [(train_vgg, "nih_vgg_v1"), (train_vgg, "nih_vgg_v2")]:
    model = f(1, df_train['path'], df_train['Finding Labels'], df_valid['path'], df_valid['Finding Labels'], mname)
    torch.save(model.state_dict(), "./{}_final.model".format(mname))
    del model
    torch.cuda.empty_cache() # PyTorch thing
