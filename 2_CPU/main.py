#%%
import argparse
import json
import numpy as np
import torch
import torch.multiprocessing
from sklearn.metrics import accuracy_score

from data_loader import get_data
from utils import select_model, select_model_amm 

def split(data_loader):
    all_data, all_targets = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        all_data.append(data)
        all_targets.append(target)
    all_data_tensor = torch.cat(all_data, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    return all_data_tensor, all_targets_tensor

def get_predictions(scores):
    """Convert continuous scores to binary predictions."""
    return np.argmax(scores, axis=1)

VAL_SPLIT = 0

def run_experiment_mask(model, dataset, ncodebook, kcentroid, config_file): 
    torch.manual_seed(0)

    train_loader, test_loader, num_classes, num_channels = get_data('/data3/neelesh/CV_Datasets', dataset, VAL_SPLIT)
    train_data, train_target = split(train_loader)
    test_data, test_target = split(test_loader)
    
    base_model = select_model(model)()
    base_model.load_state_dict(torch.load(f'../0_RES/1_NN/{model}-{dataset}.pth', map_location=torch.device('cpu')))
    base_model.eval()

    model_amm = select_model_amm(model)(base_model.state_dict(), ncodebook, kcentroid)
    
    train_fc1_target, train_fc2_target = base_model(train_data)
    test_fc1_target, test_fc2_target = base_model(test_data)
    
    print("-- Starting Training -- ") 
    train_res_amm = model_amm.forward_train(train_data, np.asarray(train_fc1_target.detach().numpy()), np.asarray(train_fc2_target.detach().numpy())) 
    
    print("-- Starting Evaluation -- ")
    test_res_amm = model_amm.forward_eval(test_data)
    
    # get NN accuracy
    train_pred = get_predictions(train_fc2_target.detach().numpy())
    test_pred = get_predictions(test_fc2_target.detach().numpy())
    
    train_accuracy = accuracy_score(train_target, train_pred)
    test_accuracy = accuracy_score(test_target, test_pred)
    
    print(f'NN - Train accuracy: {train_accuracy}')
    print(f'NN - Test accuracy: {test_accuracy}')
    
    # get AMM accuracy
    train_pred_amm = get_predictions(train_res_amm)
    test_pred_amm = get_predictions(test_res_amm)
    
    train_accuracy_amm = accuracy_score(train_target, train_pred_amm)
    test_accuracy_amm = accuracy_score(test_target, test_pred_amm)
    
    print(f'AMM - Train accuracy: {train_accuracy_amm}')
    print(f'AMM - Test accuracy: {test_accuracy_amm}')
    
    ''' 
    # get layerwise MSE of intermediate representations
    train_mse = []
    test_mse = []
    
    for i in range(len(base_intermediate_train)):
        train_mse.append(((base_intermediate_train[i] - amm_intermediate_train[i])**2).mean())
        test_mse.append(((base_intermediate_test[i] - amm_intermediate_test[i])**2).mean())
        
    print(f'-- Train MSE --')
    for i, mse in enumerate(train_mse):
        print(f'Layer {i}: {mse:.4f}')
        
    print(f'-- Test MSE --')
    for i, mse in enumerate(test_mse):
        print(f'Layer {i}: {mse:.4f}')
        
    # save results
    with open(config_file, 'r') as file:
        config = json.load(file)
    config['nn_train_accuracy'] = float(train_accuracy)
    config['nn_test_accuracy'] = float(test_accuracy)
    config['amm_train_accuracy'] = float(train_accuracy_amm)
    config['amm_test_accuracy'] = float(test_accuracy_amm)
    config['train_mse'] = [float(x) for x in train_mse]
    config['test_mse'] = [float(x) for x in test_mse]
    with open(config_file, 'w') as file:
        json.dump(config, file, indent=4)
    '''
#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Model abbreviation')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config number for configs inside of /configs')
    args = parser.parse_args()
    
    config_file = f'configs/config_{args.config}.json'
    
    with open(config_file, 'r') as file:
        config = json.load(file)
    ncodebook = config['s_subspaces']
    kcentroid = config['k_prototypes']
    
    print("-- SUBSPACE VALUES --")
    print(ncodebook)
    print("-- PROTOTYPE VALUES --")
    print(kcentroid)
    
    ncodebook = [pow(2, x) for x in ncodebook]  
    kcentroid = [pow(2, x) for x in kcentroid]
    
    # run the experiment
    run_experiment_mask(args.model, args.dataset, ncodebook, kcentroid, config_file)
    
if __name__ == "__main__":
    main()
