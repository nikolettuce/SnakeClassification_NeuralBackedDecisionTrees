import json
from datetime import datetime
import os

def write_model_to_json(loss_train, acc_train, fs_train, loss_val, acc_val, fs_val, n_epochs, model_name,fp):
    '''
    write model performance to a json file named
    DATE_EPOCHS_ MODELNAME_performance.json
    '''
    data = {}
    
    # train performance
    data['loss_train'] = loss_train
    data['acc_train'] = acc_train
    data['fs_train'] = fs_train
    
    # validation performance
    data['loss_val'] = loss_val
    data['acc_val'] = acc_val
    data['fs_val'] = fs_val
    
    # date
    now = datetime.now()
    now_formatted = now.strftime("%d%m%Y_%H:%M")
    
    # format filename
    file_name = "{}_{}_{}_performance.json".format(now_formatted, n_epochs, model_name)
    write_fp = os.path.join(fp, file_name)
    
    # write to file
    with open(write_fp, 'w') as json_file:
        json.dump(data, json_file)
    
    print('Wrote model performance to {}'.format(write_fp))