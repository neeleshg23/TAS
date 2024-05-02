import torch
from models.resnet import resnet18
import pandas as pd
import json
from flopth import flopth
import numpy as np

# model = resnet18(10,3)
# inp = torch.rand(1,3,32,32)

# flops, params = flopth(model, inputs=(inp,), show_detail=True, bare_number=True) 
# print(flops, params)

data = {
    'module_name': ['conv1', 'bn1', 'layer1.0.residual_function.0', 'layer1.0.residual_function.1', 'layer1.0.residual_function.2', 'layer1.0.residual_function.3', 'layer1.0.residual_function.4', 'layer1.0.shortcut.0', 'layer1.0.shortcut.1', 'layer1.1.residual_function.0', 'layer1.1.residual_function.1', 'layer1.1.residual_function.2', 'layer1.1.residual_function.3', 'layer1.1.residual_function.4', 'layer1.1.shortcut', 'layer2.0.residual_function.0', 'layer2.0.residual_function.1', 'layer2.0.residual_function.2', 'layer2.0.residual_function.3', 'layer2.0.residual_function.4', 'layer2.0.shortcut.0', 'layer2.0.shortcut.1', 'layer2.1.residual_function.0', 'layer2.1.residual_function.1', 'layer2.1.residual_function.2', 'layer2.1.residual_function.3', 'layer2.1.residual_function.4', 'layer2.1.shortcut', 'layer3.0.residual_function.0', 'layer3.0.residual_function.1', 'layer3.0.residual_function.2', 'layer3.0.residual_function.3', 'layer3.0.residual_function.4', 'layer3.0.shortcut.0', 'layer3.0.shortcut.1', 'layer3.1.residual_function.0', 'layer3.1.residual_function.1', 'layer3.1.residual_function.2', 'layer3.1.residual_function.3', 'layer3.1.residual_function.4', 'layer3.1.shortcut', 'layer4.0.residual_function.0', 'layer4.0.residual_function.1', 'layer4.0.residual_function.2', 'layer4.0.residual_function.3', 'layer4.0.residual_function.4', 'layer4.0.shortcut.0', 'layer4.0.shortcut.1', 'layer4.1.residual_function.0', 'layer4.1.residual_function.1', 'layer4.1.residual_function.2', 'layer4.1.residual_function.3', 'layer4.1.residual_function.4', 'layer4.1.shortcut', 'adaptive_pool', 'linear'],
    'module_type': ['Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sequential', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Conv2d', 'BatchNorm2d', 'Sequential', 'AdaptiveAvgPool2d', 'Linear'],
    'in_channels': [3, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 128, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512],
    'out_channels': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 10],
    'H': [32, 16, 16, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'W': [32, 16, 16, 8, 8, 8, 8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'F': [3, 0, 3, 0, 0, 3, 0, 1, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 1, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 1, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 0],
    'P': [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'S': [2, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'FLOPS': [442368, 32768, 2359296, 8192, 4096, 2359296, 8192, 262144, 8192, 2359296, 8192, 4096, 2359296, 8192, 0, 1179648, 4096, 2048, 2359296, 4096, 131072, 4096, 2359296, 4096, 2048, 2359296, 4096, 0, 1179648, 2048, 1024, 2359296, 2048, 131072, 2048, 2359296, 2048, 1024, 2359296, 2048, 0, 1179648, 1024, 512, 2359296, 1024, 131072, 1024, 2359296, 1024, 512, 2359296, 1024, 0, 512, 5120]
}

df = pd.DataFrame(data)

iter_json = 0

json_data = {
    "num_layers": 22,
    "subspaces": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "prototypes": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Create new columns 'N' and 'K' with default value 0
df['N_sub'] = 0
df['K_prot'] = 0

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    if row['module_type'] == 'Conv2d' or row['module_type'] == 'Linear':
        # Assign corresponding values from JSON to 'N' and 'K' columns
        df.at[index, 'N_sub'] = json_data['subspaces'][iter_json]
        df.at[index, 'K_prot'] = json_data['prototypes'][iter_json]
        iter_json += 1
        
df['FLOPS_ADJ'] = 0

for index, row in df.iterrows():
    if row['module_type'] == 'Conv2d':
        H_prime = (row['H'] - row['F'] + 2 * row['P']) / row['S'] + 1
        W_prime = (row['W'] - row['F'] + 2 * row['P']) / row['S'] + 1
        table_lookup_scalar = (pow(2, row['N_sub']) * row['K_prot'] + row['out_channels'] * row['N_sub'])
        df.at[index, 'FLOPS_ADJ'] = H_prime * W_prime * table_lookup_scalar 
    elif row['module_type'] == 'Linear':
        df.at[index, 'FLOPS_ADJ'] = row["in_channels"] * row['K_prot'] + row['out_channels'] * row['N_sub']
    else:
        df.at[index, 'FLOPS_ADJ'] = row['FLOPS']    

print(df)

print("Intitial FLOPs: ", df['FLOPS'].sum())    
print("Total FLOPs: ", df['FLOPS_ADJ'].sum())   