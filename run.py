# import custom modules
import sys
sys.path.insert(0, "src/util")
sys.path.insert(0, "src/model")
sys.path.insert(0, "src/data_util")

# imports for model
import torch
import torchvision
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

from baseline import *

from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss

from wn_utils import *
from graph import *
from dir_grab import *
from hierarchy import *
from debug_data import *
from write_to_json import *
from loss import *

from datetime import datetime

def main(targets):
    '''
    runs project code based on targets
    
    configure filepaths based on data-params.json
    
    targets:
    data - builds data from build.sh, will run if no model folder is found.
    test - checks if data target has been run, runs model to train
    hierarchy - creates induced hierarchy and visualizes it
    '''
    
    if 'data' in targets:
        print('---> Running data target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
        
        # check for directory's existence and rename, raise if no directory exists
        DATA_DIR = data_cfg['dataDir']
        if os.path.isdir(os.path.join(DATA_DIR, 'train')): # default name after extraction
            os.rename(
                os.path.join(DATA_DIR, 'train'),
                os.path.join(DATA_DIR, 'train_snakes_r1')
            )
        elif not os.path.isdir(os.path.join(DATA_DIR, 'train')) | os.path.isdir(os.path.join(DATA_DIR, 'train_snakes_r1')):
            raise Exception('Please run build.sh before running run.py')
        
        # important name variables
        TRAIN_DIR = os.path.join(DATA_DIR, 'train_snakes_r1')
        VALID_DIR = os.path.join(DATA_DIR, 'valid_snakes_r1') # new dir to be made
        train_pct = 0.8
        
        # delete corrupted data from download
        delete_corrupted(TRAIN_DIR)

        # create validation set
        create_validation_set(DATA_DIR, TRAIN_DIR, VALID_DIR, train_pct)
        
        print("---> Finished running data target.")
        
    if 'train' in targets:
        print('---> Running train target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
            
        # check that data target has been ran
        VALID_DIR = os.path.join(data_cfg['dataDir'], 'valid_snakes_r1')
        if not os.path.isdir(VALID_DIR):
            raise Exception('Please run data target before running test')
        
        if 'SoftTreeSupLoss' in targets:
            criterion = SoftTreeLoss_wrapper(data_cfg)
        elif 'HardTreeSupLoss' in targets:
            criterion = HardTreeLoss_wrapper(data_cfg)
        else:
            criterion = nn.CrossEntropyLoss()
        
        
        #TESTING
        torch.autograd.set_detect_anomaly(True)
        
        # create and train model
        model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = run_model(data_cfg, model_cfg, criterion)
        
        # write performance to data/model_logs
        write_model_to_json(
            loss_train,
            acc_train,
            fs_train,
            loss_val,
            acc_val,
            fs_val,
            n_epochs = model_cfg['nEpochs'],
            model_name = model_cfg['modelName'],
            fp = model_cfg['performancePath']
        )
        
        print("---> Finished running test target.")
        
    if 'test' in targets:
        print('---> Running test target...')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
        
        # check that data target has been ran
        VALID_DIR = os.path.join(data_cfg['dataDir'], 'valid_snakes_r1')
        if not os.path.isdir(VALID_DIR):
            raise Exception('Please run data target before running test')
        
        # Loss function
        criterion = SoftTreeLoss_wrapper(data_cfg)
        
        #TESTING
        torch.autograd.set_detect_anomaly(True)
        
        # create and train model
        model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = run_model(data_cfg, model_cfg, criterion)
        
        # write performance to data/model_logs
        write_model_to_json(
            loss_train,
            acc_train,
            fs_train,
            loss_val,
            acc_val,
            fs_val,
            n_epochs = model_cfg['nEpochs'],
            model_name = model_cfg['modelName'],
            fp = model_cfg['performancePath']
        )
        
        print("---> Finished running train target.")
        
    if "hierarchy" in targets:
        print('---> Runnning hierarchy target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
            
        # use pretrained densenet
        model = models.densenet121(pretrained=True)
        # set features from classes, in this case 45, input_size always 224
        model.classifier = nn.Linear(model.classifier.in_features, model_cfg['nClasses'])
        input_size = model_cfg['inputSize']
        
        ## load state dict from previous
        #if not os.path.exists(data_cfg['hierarchyModelPath']):
        #    raise Exception('Please run train target before hierarchy target, or change hierarchyModelPath in data-params if model has been trained.')
        #model_weights = torch.load(data_cfg['hierarchyModelPath'])
        #model.load_state_dict(model_weights)
        
        # generate hierarchy
        print("---> Generating hierarchy...")
        generate_hierarchy(
            dataset='snakes',
            arch = data_cfg['hierarchyModel'],
            model = model,
            method = 'induced'
        )
        print("---> Finished generating hierarchy.")
        
        # test hierarchy
        print("---> Testing hierarchy...")
        test_hierarchy(
            'snakes',
            os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON'])
        )
        
        generate_hierarchy_vis(
            os.path.join(data_cfg['hierarchyPath'], data_cfg['hierarchyJSON']),
            'snakes'
        )
        
    if "nbdt_loss" in targets:
        print('---> Runnning nbdt_loss target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
        
        # create dataloaders
        dataloaders_dict, num_classes = create_dataloaders(
            data_cfg['dataDir'],
            model_cfg['batchSize'],
            model_cfg['inputSize']
        )
        
        # Detect if we have a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model for this run
        model, input_size = initialize_model(
            model_cfg['modelName'],
            num_classes,
            feature_extract = model_cfg['featureExtract'],
            use_pretrained = True
        )
        
        #model = model.to(device) # make model use GPU
        
        # set features from classes, in this case 45, input_size always 224
        model.classifier = nn.Linear(model.classifier.in_features, model_cfg['nClasses'])
        model_weights = torch.load(data_cfg['hierarchyModelPath'])
        
        print('---> NBDT transition beginning...')
        if 'softloss' in targets:
            criterion = SoftTreeLoss_wrapper(data_cfg)
        elif 'hardloss' in targets:
            criterion = HardTreeLoss_wrapper(data_cfg)
        
        # using induced hierarchy, create model 
        nbdt_model = SoftNBDT(
            model = model,
            dataset = 'snakes', 
            hierarchy='induced-densenet121',
            path_graph = "./data/hierarchies/snakes/graph-induced-densenet121.json",
            path_wnids = "./data/wnids/snakes.txt"
        )
        print('---> NBDT transition finished.')
        
        print('---> Begin inference testing...')
        # iterate over data
        for inputs, labels in dataloaders_dict['valid_snakes_r1']:
            #inputs = inputs.to(device)
            # labels = labels.to(device)

            # calculate loss from model outputs
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # statistics
            labels_cpu = labels.cpu().numpy()
            predictions_cpu = preds.cpu().numpy()
            Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
            fscore.append(Fscore)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        print('---> Finished inference testing.')
        
        # calculate final stats
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        epoch_fscore = np.average(np.array(fscore))

        print('{} Loss: {:.4f} Acc: {:.4f} F: {:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_fscore))
        
    if "baseline_cnn" in targets:
        print('---> Runnning baseline_cnn target')
        
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
            print('---> loaded data config')
            
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)
            print('---> loaded model config')
        
        # check that data target has been ran
        VALID_DIR = os.path.join(data_cfg['dataDir'], 'valid_snakes_r1')
        if not os.path.isdir(VALID_DIR):
            raise Exception('Please run data target before running test')
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # create and train model
        model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = run_model(data_cfg, model_cfg, criterion)
        
        # write performance to data/model_logs
        write_model_to_json(
            loss_train,
            acc_train,
            fs_train,
            loss_val,
            acc_val,
            fs_val,
            n_epochs = model_cfg['nEpochs'],
            model_name = model_cfg['modelName'],
            fp = model_cfg['performancePath']
        )
        
        print("---> Finished running baseline_cnn target.")
        
        
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)