import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.dlpack as dlpack
from tqdm import tqdm
import numpy as np

from .amm.pq_amm_cnn import PQ_AMM_CNN
from .amm.vq_amm import PQMatmul


def im2col(input_data, kernel_size, stride, pad):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - kernel_size) // stride + 1
    out_w = (W + 2*pad - kernel_size) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    img = np.asarray(img)
    col = np.zeros((N, C, kernel_size, kernel_size, out_h, out_w))

    for y in range(kernel_size):
        y_max = y + stride*out_h
        for x in range(kernel_size):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

class Net_AMM:
    def __init__(self, state_dict, ncodebook=[1]*5, kcentroid=[4]*5):
        self.n = ncodebook
        self.k = kcentroid
        
        self.state_dict = state_dict
        
        self.conv1_weights = state_dict['conv1.weight'].numpy()
        self.conv1_bias = state_dict['conv1.bias'].numpy()
        self.conv2_weights = state_dict['conv2.weight'].numpy()
        self.conv2_bias = state_dict['conv2.bias'].numpy()
        self.conv3_weights = state_dict['conv3.weight'].numpy()
        self.conv3_bias = state_dict['conv3.bias'].numpy()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1_weights = state_dict['fc1.weight'].numpy()
        self.fc1_bias = state_dict['fc1.bias'].numpy()
        self.fc2_weights = state_dict['fc2.weight'].numpy()
        self.fc2_bias = state_dict['fc2.bias'].numpy()
        
        self.amm_estimators = []
        self.amm_queue = []
        
    def conv2d(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T
        
        col_W = np.asarray(col_W) 
        out = np.dot(col, col_W) + b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def conv2d_amm(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])
        
        est = PQ_AMM_CNN(self.n.pop(0), self.k.pop(0))  
        est.fit(col_matrix_2d, col_W)

        est.reset_for_new_task()
        est.set_B(col_W)
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        self.amm_estimators.append(est)
        
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out
    
    def conv2d_eval(self, est, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])
        
        est.reset_enc()
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out

    def relu(self, x):
        return np.maximum(0, x)
    
    def linear_amm(self, input_data, weights, bias, target):
        target = torch.from_numpy(target).float()
        input_data = torch.from_numpy(input_data).float()
        weights, bias = self.fine_tune_fc_layer(input_data, weights, bias, target, epoch=300, lr=0.001)
        est = PQMatmul(self.n.pop(0), self.k.pop(0))
        input_data = input_data.detach().numpy()
        est.fit(input_data, weights)
        est.reset_for_new_task()
        est.set_B(weights)
        res = est.predict(input_data, weights) + bias
        self.amm_estimators.append(est)
        return res
    
    def linear_eval(self, est, input_data, weights, bias):
        est.reset_enc()
        weights, bias = np.asarray(weights), np.asarray(bias) 
        res = est.predict(input_data, weights) + bias
        return res
    
    def fine_tune_fc_layer(self, new_input, weight, bias, target, epoch=300, lr=0.001):
        
        linear_layer = nn.Linear(weight.shape[0], weight.shape[1])
       
        weight_torch = torch.from_numpy(weight).float().t() # also transpose the weight matrix 
        bias_torch = torch.from_numpy(bias).float()
         
        with torch.no_grad():
            linear_layer.weight.copy_(weight_torch)  
            linear_layer.bias.copy_(bias_torch)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(linear_layer.parameters(), lr=lr)

        for i in tqdm(range(epoch)):
            optimizer.zero_grad()
            new_output = linear_layer(new_input)
            loss = criterion(new_output, target)
            loss.backward()
            optimizer.step()
            if loss.item() < 1e-5:
                break
        
        new_weight, new_bias = linear_layer.weight.detach().numpy(), linear_layer.bias.detach().numpy()
        return new_weight.T, new_bias  # Transpose back the weight matrix
        
    def dropout(self, input, p):
        binary_value = np.random.rand(*input.shape) > p
        res = input * binary_value
        res /= p
        return res
    
    def forward(self, x):
        intermediate_results = []
        out = self.conv2d(x, self.conv1_weights, self.conv1_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        out = self.conv2d(out, self.conv2_weights, self.conv2_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        out = self.conv2d(out, self.conv3_weights, self.conv3_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        # flatten image input
        out = out.reshape(out.shape[0], -1)
        
        out = np.dot(out, self.fc1_weights.T) + self.fc1_bias   
        intermediate_results.append(out)
        out = self.relu(out)
        
        # out = self.dropout(out, 0.25)
        
        out = np.dot(out, self.fc2_weights.T) + self.fc2_bias
        intermediate_results.append(out)
        
        return out, intermediate_results
    
    def forward_train(self, x, target1, target2):
        intermediate_results = []

        out = self.conv2d_amm(x, self.conv1_weights, self.conv1_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float() 
        out = self.pool(out)
        out = out.detach().numpy()
        
        out = self.conv2d_amm(out, self.conv2_weights, self.conv2_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        out = self.conv2d_amm(out, self.conv3_weights, self.conv3_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        # flatten image input
        out = out.reshape(out.shape[0], -1)
        
        out = self.linear_amm(out, self.fc1_weights.T, self.fc1_bias, target1)
        intermediate_results.append(out)
        out = self.relu(out)
        
        # out = self.dropout(out, 0.25)
        
        out = self.linear_amm(out, self.fc2_weights.T, self.fc2_bias, target2)
        intermediate_results.append(out)
        
        return out, intermediate_results
        
    def forward_eval(self, x):
        intermediate_results = []
        
        est = self.amm_estimators.pop(0)
        out = self.conv2d_eval(est, x, self.conv1_weights, self.conv1_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float() 
        out = self.pool(out)
        out = out.detach().numpy()
        
        est = self.amm_estimators.pop(0) 
        out = self.conv2d_eval(est, out, self.conv2_weights, self.conv2_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        est = self.amm_estimators.pop(0) 
        out = self.conv2d_eval(est, out, self.conv3_weights, self.conv3_bias, pad=1)
        intermediate_results.append(out)
        out = self.relu(out)
        out = torch.from_numpy(out).float()
        out = self.pool(out)
        out = out.detach().numpy()
        
        # flatten image input
        out = out.reshape(out.shape[0], -1)
        
        est = self.amm_estimators.pop(0) 
        out = self.linear_eval(est, out, self.fc1_weights.T, self.fc1_bias)
        intermediate_results.append(out)
        out = self.relu(out)
        
        # out = self.dropout(out, 0.25)
        
        est = self.amm_estimators.pop(0) 
        out = self.linear_eval(est, out, self.fc2_weights.T, self.fc2_bias)
        intermediate_results.append(out)
        
        return out, intermediate_results