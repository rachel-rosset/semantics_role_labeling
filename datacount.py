import itertools
import json

file_path = "data/agent/train.json"
dataset = {}
is_divider = lambda line: line.strip() == ''
with open(file_path, 'r') as f:
    dataset = json.load(f)
    for data in dataset:
        print(dataset[data])


print("Length: " , len(dataset.keys()))