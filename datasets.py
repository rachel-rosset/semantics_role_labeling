from decomp import UDSCorpus 
from collections import defaultdict
from tqdm import tqdm

import json
import os

ROLES = {#"agent": lambda p: ((p['volition']['value'] > 0) or p['instigation']['value'] > 0) and (p['existed_before']['value'] > 0),
         "patient": lambda b: ((b['change_of_state']['value'] > 0) or b['change_of_state_continuous']['value'] > 0) and (b['existed_before']['value'] > 0),
         "location": lambda c: (c['location']['value'] > 0),
         "instrument": lambda d: (d['was_used']['value'] > 0) and (d['existed_during']['value'] > 0),
        }

PATHNAME = "/Users/rrosset/Documents/Spring_2021/EventSemantics/hw1/data"


def parse_node_name(node):
    type_, idx = node.split('-')[-2:]
    return type_, idx

def parse_edge_name(edge):
    pred_head_idx = None 
    arg_head_idx = None

    type_, idx = parse_node_name(edge[0])

    if type_ == 'pred':
        pred_head_idx = idx
    elif type_ == 'arg':
        arg_head_idx = idx
    else:
        raise ValueError(f"{edge[0]}")

    type_, idx = parse_node_name(edge[1])

    if type_ == 'pred':
        pred_head_idx = idx
    elif type_ == 'arg':
        arg_head_idx = idx
    else:
        raise ValueError(f"{edge[1]}")

    assert ((pred_head_idx != None) and (arg_head_idx != None))

    return pred_head_idx, arg_head_idx


def process_split(split, role, criteria):
    dataset = {}
    count_pos = 0
    count_neg = 0
    for graphid, graph in split.items():
        tokens = tuple(graph.sentence.split())
        #print('tokens', tokens)
        semantic_edges = graph.semantics_edges()
        for edge, properties in semantic_edges.items():
            #print("props", properties)
            if 'protoroles' in properties:
                #print("HERE WE MADE IT AAAAA")
                try:
                    pred_head_idx, arg_head_idx = parse_edge_name(edge)
                except:
                    import pdb; pdb.set_trace()

                try:
                    role_val = criteria(properties['protoroles'])
                    if role_val == True:
                        label = "positive"
                    else:
                        label = "negative"
                    item_id = "|||".join([graphid, pred_head_idx, arg_head_idx])
                    dataset[item_id] = {"graphid": graphid,
                                        "tokens": tokens,
                                        "predicate_head_idx": pred_head_idx,
                                        "argument_head_idx": arg_head_idx,
                                        "label": label}
                    if role_val == True:
                        count_pos += 1
                    else:
                        count_neg +=1
                except:
                    continue
        
    print("\n POSITIVE: %d AND NEG: %d \n", count_pos, count_neg)
    return dataset


def main():
    splits = {  "train": UDSCorpus(split='train'),
                "dev": UDSCorpus(split='dev'),
                "test": UDSCorpus(split='test')}
    #uds_train = UDSCorpus(split='train')
    #short_train = {k: uds_train[k] for k in list(uds_train)[:1000]}

    datasets = defaultdict(dict)

    for role, criteria in tqdm(ROLES.items()):
        for split in tqdm(['train', 'dev', 'test']):
            datasets[role][split] = process_split(splits[split], role, criteria)
            print(len(datasets[role][split]))

    for role in datasets.keys():
        role_path = os.path.join(PATHNAME, role)
        if not os.path.exists(role_path):
            os.mkdir(role_path)

        for split in datasets[role].keys():
            with open(os.path.join(role_path, f"{split}.json"), "w") as f:
             json.dump(datasets[role][split], f, indent=2)



if __name__ == "__main__":
    main()