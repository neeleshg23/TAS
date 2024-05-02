import csv
import json

with open('input_data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    next(csv_reader)
    
    for row in csv_reader:
        config_num = row[1]
        
        s_subspaces = [int(value) for value in row[2:24]]
        k_prototypes = [int(value) for value in row[24:]]
        
        json_content = {
            's_subspaces': s_subspaces,
            'k_prototypes': k_prototypes
        }
        
        json_file_name = f'config_{config_num}.json'
        
        with open(json_file_name, 'w') as json_file:
            json.dump(json_content, json_file)
        
        print(f'Generated {json_file_name}')