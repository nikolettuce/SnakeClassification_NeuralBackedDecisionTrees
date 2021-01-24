# imports
import torch
import torchvision
import random
import matplotlib.pyplot as plt
import os
import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
from PIL import Image

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    '''
    trains model by using train and validation sets
    '''
    # define lists
    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    
    loss_train_evo=[]
    acc_train_evo=[]
    fs_train_evo=[]
    
    loss_val_evo=[]
    acc_val_evo=[]
    fs_val_evo=[]
    
    total_train=round(47626/batch_size)
    
    
    for epoch in range(num_epochs):
        i = 0
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # determine if in train or validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            fscore = []

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients before beginning backprop
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    """
                    if i==round(0.25*total_train):
                        print('Forward Passed 25%')
                    if i==round(0.5*total_train):
                        print('Forward Passed 50%')
                    if i==round(0.75*total_train):
                        print('Forward Passed 75%')
                    i = i + 1
                    """
                    # calculate loss from model outputs
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                labels_cpu = labels.cpu().numpy()
                predictions_cpu = preds.cpu().numpy()
                Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
                fscore.append(Fscore)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_fscore = np.average(np.array(fscore))
            
            print('{} Loss: {:.4f} Acc: {:.4f} F: {:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_fscore))
            
            if phase == 'train':
                loss_train_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_train_evo.append(epoch_acc)
                fs_train_evo.append(epoch_fscore)                
            else:
                loss_val_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_val_evo.append(epoch_acc)
                fs_val_evo.append(epoch_fscore) 
                
            # deep copy the model
            if phase == 'val' and epoch_fscore > best_fscore:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_train_evo, acc_train_evo, fs_train_evo, loss_val_evo, acc_val_evo, fs_val_evo

def set_parameter_requires_grad(model, feature_extracting):
    '''
    sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    '''
    initializes pretrained vgg model
    '''
    model_ft = None
    input_size = 0

    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size
