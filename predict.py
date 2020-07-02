from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import Configuration
import subprocess
from pathlib import Path
import pickle
import sys, os
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from datetime import datetime
import numpy as np
import random
import torch
import shutil
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
from torch.nn import Parameter
import math
import time
import re
import copy
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from utils import preprocess_input
from models import BertMultitaskClassifier
import pdb
from optimization import *
import logging
import time
import hashlib
import re
from bisect import bisect_right, bisect_left

# torch.autograd.set_detect_anomaly(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)

# logger = logging.getLogger(__name__)
def main(args, tmp_file_dir):

    data_dir = args.data_dir
    data_test = pickle.load(open(os.path.join(args.data_dir, args.test_pkl), 'rb'))
    
    test_kg_datas = None
    if args.use_knowledge:
        test_kg_datas = pickle.load(open(f'{args.data_dir}/{args.test_kg_datas}.pkl','rb'))

    args._label_to_id_t = OrderedDict([('None', 0), ('Gene_expression', 1), ('Localization', 2), ('Transcription', 3), ('Binding', 4), ('Phosphorylation', 5), ('Positive_regulation', 6), ('Regulation', 7), ('Protein_catabolism', 8), ('Protein', 9), ('Negative_regulation', 10)])
    args._id_to_label_t = {0: 'None', 1: 'Gene_expression', 2: 'Localization', 3: 'Transcription', 4: 'Binding', 5: 'Phosphorylation', 6: 'Positive_regulation', 7: 'Regulation', 8: 'Protein_catabolism', 9: 'Protein', 10: 'Negative_regulation'}
    args._label_to_id_i = OrderedDict([('None', 0), ('Theme', 1), ('Cause', 2)])
    args._id_to_label_i = {0: 'None', 1: 'Theme', 2: 'Cause'}
    args.SIMPLE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization', 'Phosphorylation']
    args.REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
    args.BIND = ['Binding']


    test_input_ids = torch.tensor(data_test['tokenized_ids'], dtype=torch.long)
    test_input_masks = torch.tensor(data_test['mask_ids'], dtype=torch.long)
    test_segment_ids = torch.tensor(data_test['segment_ids'], dtype=torch.long)
    test_entity_labels = torch.tensor([ [ args._label_to_id_t[entity_label] for entity_label in entity_labels]  for entity_labels in data_test['entity_labels'] ], dtype=torch.long)
    test_sample_ids = torch.tensor(data_test['sample_ids'], dtype=torch.long)
    
    test_data = TensorDataset(test_input_ids, test_input_masks, test_segment_ids, test_entity_labels, test_sample_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    
    test_doc_ids = data_test['doc_ids']
    test_abs_spans = data_test['abs_spans']

    if 'scibert' in args.model:
        bert_weights_path = 'allenai/scibert_scivocab_uncased'
            
    elif 'biobert' in args.model:
        bert_weights_path= 'bert_weights/biobert_v1.1_pubmed'
    elif 'bert' in args.model:
        bert_weights_path=args.model


    if args.do_test:
        model = BertMultitaskClassifier(args, bert_weights_path=bert_weights_path)
        
        output_model_file = os.path.join(args.model_dir, "pytorch_model.bin")
        
        if torch.cuda.is_available():
            checkpoint = torch.load(output_model_file)
        else:
            checkpoint = torch.load(output_model_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        gold=False
        y_trues_e, y_preds_e, y_trues_r, y_preds_r, data_out = model.predict(test_dataloader, gold, args, test=True, eval_kg_datas=test_kg_datas)
         

        merged_pkl = f'{tmp_file_dir}/{args.out_test_pkl}'
        write_pkl(args, data_out, test_abs_spans, test_doc_ids, merged_pkl, gold_tri=False)
        unmerge_normalize(args, tmp_file_dir, args.output_dir.split('/')[-1], merged_pkl)

    
def unmerge_normalize(args, tmp_file_dir, model_name, merged_pkl):
    '''
    Call unmerge & normalize on test set
    '''

    
    unmerge_cmd=f'python unmerg_write.py -pred_pkl={merged_pkl}'\
    f' -protIdBySpan={tmp_file_dir}/{args.doc_id}_protIdBySpan.pkl'\
    f' -origIdById={tmp_file_dir}/{args.doc_id}_origIdById.pkl'\
    f' -out_dir={tmp_file_dir}'\
    f' -input_dir={tmp_file_dir}'
    
    p = subprocess.Popen(unmerge_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    
    # normalization
    normalize_cmd = f'./eval/tools/a2-normalize.pl -g {tmp_file_dir}/ -u {tmp_file_dir}/*.a2'
    p = subprocess.Popen(normalize_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    # copy a1 & txt files to a2
    # [ shutil.copy(source_file, output_dir) for file in os.listdir(output_dir) for source_file in [f'genia_cord_19/{file.split(".a2")[0]}.txt', f'genia_cord_19/{file.split(".a2")[0]}.a1']]
        
            
    

def write_pkl(args, data_out, abs_spans, doc_ids, out_file, gold_tri=True):
    '''
    data_out = {'doc_ids':all_doc_ids, 
        'predicted_entities':preachine Learindicted_entities,  
        'predicted_interactions': predicted_interactions, 
        'predicted_interaction_labels':predicted_interaction_labels,
        'gold_entities': all_gold_entities,
        'gold_interactions': all_gold_interactions,
        'gold_interaction_labels': all_gold_interaction_labels,
        'input_ids':all_input_ids

        }
    '''
    outs = []

        
    predicted_etities = data_out['predicted_entities']
    predicted_interaction_labels = data_out['predicted_interaction_labels']
    predicted_interactions = data_out['predicted_interactions']
    gold_entities = data_out['gold_entities']
    gold_interactions = data_out['gold_interactions']
    input_ids = data_out['input_ids']

    
    # predicted_etities = np.vstack(predicted_etities)
    # convert to categorical labels
    gold_entities = [ np.array([args._id_to_label_t[entity] for entity in entities], dtype=object) for entities in  gold_entities]
    predicted_etities = [np.array([args._id_to_label_t[entity] for entity in entities], dtype=object) for entities in  predicted_etities]

    assert len(predicted_interactions) == len(predicted_interaction_labels) == len(predicted_interactions) == \
         len(gold_entities)  == len(abs_spans) == len(doc_ids)\
        , (len(predicted_interactions), len(predicted_interaction_labels), len(predicted_interactions), len(gold_entities), len(abs_spans), len(doc_ids) )

    
    outs = []
    for i in range(len(gold_entities)):
            
        if gold_tri:
            out = tuple([doc_ids[i], input_ids[i], None, gold_entities[i], predicted_interactions[i], predicted_interaction_labels[i], abs_spans[i], None, None])
        else:
            out = tuple([doc_ids[i], input_ids[i], None, predicted_etities[i], predicted_interactions[i], predicted_interaction_labels[i], abs_spans[i], None, None])
        assert len(abs_spans[i]) == len(gold_entities[i]), (len(abs_spans[i]), len(gold_entities[i]))
        
        outs.append(out)
    
    with open(out_file, 'wb') as f:
        pickle.dump(outs, f, protocol=4)
    # print('out pickle file saved as {}'.format(out_file))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_json_output(tmp_file_dir, doc_id):

    '''
    Read in a1, a2, and char2token_map files and generate output.
    returns:
        res: a list of dictionary. Keys: {"tokens": [], "events": [], "ner": [[]]}
    '''

    def same_event(event1, event2):
        return event1['triggers'] == event2['triggers'] and event1['event_type'] == event2['event_type'] and event1['arguments'] == event2['arguments']

    res = [{}]
    with open(f'{tmp_file_dir}/{doc_id}.a1','r') as f:
        lines = [re.split('\s', line.strip()) for line in f.readlines()]
    try:
        with open(f'{tmp_file_dir}/{doc_id}.a2','r') as f:
            lines += [re.split('\s', line.strip()) for line in f.readlines()]
    except:
        print("no event predicted")
    
    with open(f'{tmp_file_dir}/{doc_id}_preprocess_result.pkl','rb') as f:
        preprocess_result = pickle.load(f)
    
    # order the events such that th
    entity_lines = lines 
    
    entity_map = {}

    tokens = preprocess_result['tokens']
    ner = preprocess_result['ner']
    char2doctoken_map = preprocess_result['char2doctoken_map']
    sentence_offsets = preprocess_result['sentence_offsets']

    # create a list of character offset for each first token in each sentence to determine which sentence the event belongs to
    

    # constuct output
    # for tok, ne in zip(tokens, ner):
    #     cur = {
    #         'tokens':tok,
    #         'events':[],
    #         'ner':ne
    #     }
    #     res.append(cur)
    res[0]['tokens'] = tokens
    res[0]['events'] = []
    res[0]['ner'] = ner

    # constrcut event
    for line in lines:
        # if it is an entity (triger, argument)
        if line[0].startswith('T'):
            # start, end character offset, and text
            entity_map[line[0]] = [int(line[2]), int(line[3]), ' '.join(line[4:])]
        # map event to its trigger position
        else:
            trigger_entity_id = line[1].split(':')[1]
            entity_map[line[0]] = entity_map[trigger_entity_id]
    
    # constructs events
    for line in lines:
        if line[0].startswith('E'):
            # each event is represented as a dictionary
            current_event = {}

            event_type, entity_id = line[1].split(':')
            trigger_start_char_offset = entity_map[entity_id][0]

            # find which sentence this event belongs to.
            sentence_idx = 0

            current_event['event_type'] = event_type
            current_event['triggers'] = [{'event_type':event_type,
                         'text': entity_map[entity_id][2],
                         'start_token': char2doctoken_map[list(char2doctoken_map.keys())[bisect_right(list(char2doctoken_map.keys()), entity_map[entity_id][0]) - 1]],

                         # the first -1 is for go back one character because GENIA annotation at the character level is exlusive at the end i.e. [start, end)
                         'end_token': char2doctoken_map[list(char2doctoken_map.keys())[bisect_right(list(char2doctoken_map.keys()), entity_map[entity_id][1] -1) -1]]
                          }]

            current_event['arguments'] = []
            for entity in line[2:]:

                role, entity_id  = entity.split(':')
                
                current_argumet = {
                    'role':role,
                    'text': entity_map[entity_id][2],
                    'start_token': char2doctoken_map[list(char2doctoken_map.keys())[bisect_right(list(char2doctoken_map.keys()), entity_map[entity_id][0]) - 1]],
                    # the first -1 is for go back one character because GENIA annotation at the character level is exlusive at the end i.e. [start, end)
                    'end_token': char2doctoken_map[list(char2doctoken_map.keys())[bisect_right(list(char2doctoken_map.keys()), entity_map[entity_id][1] -1) -1]]
                }
                current_event['arguments'].append(current_argumet)

            duplicate_event = False
            # make sure the current event is not the same as any other events:
            for prev_event in res[sentence_idx]['events']:
                if same_event(prev_event, current_event):
                    duplicate_event = True
                    break
            
            if not duplicate_event:
                res[sentence_idx]['events'].append(current_event)

    
    return res
    
def biomedical_evet_extraction(user_input):
    '''
    user_input: str. A biomedical corpus to run event extraction pipeline on.
    '''
    args = Configuration()
    

    # detertmine the prefix for processed data based on the model name
    if 'biobert_large' in args.model:
        prefix = 'GE11_biobert_large'
    elif 'biobert' in args.model:
        prefix = 'GE11_biobert_v1.1_pubmed'
    elif 'scibert' in args.model:
        prefix = 'GE11_scibert_scivocab_uncased'
    elif 'bert' in args.model:
        prefix = 'GE11_bert-base-uncased'
    else:
        raise NotImplementedError

    #args.eval_gold = True if args.pipe_epoch >= 1000 else False
    args.SIMPLE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization', 'Phosphorylation']
    args.REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
    args.BIND = ['Binding']

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # name the document based on time & sha256 hash append the .d0 for unmerging to work
    doc_id = hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest() + '.d0'
    args.doc_id = doc_id

    tmp_file_dir = 'tmp'
    os.makedirs(tmp_file_dir, exist_ok=True)

    # store doc_id to txt
    with open(f'{tmp_file_dir}/{doc_id}.txt','w') as f:
        f.write(user_input)
    
    
    
    args.data_dir = tmp_file_dir
    args.model_dir = "weights/pipeline_scibert_batch_4_lr_3e-5_epochs100_pepochs100_seed_42_dp0.1_know-false_kg_emb300_ent1000_rel300"

    # remove '\n' which can break the system 
    user_input = user_input.replace('\n',' ')
    

    # preprocess the data and store in 4 different files
    preprocess_input(user_input, doc_id, tmp_file_dir)

    # input & output
    args.test_pkl = f'{doc_id}.pkl'
    args.out_test_pkl = f'{doc_id}_merged.pkl'

    # run event extraction    
    main(args, tmp_file_dir)

    # read a2 and output json    
    output = create_json_output(tmp_file_dir, doc_id)
    
    # delete genereated intermediate files
    # for filename in glob.glob(tmp_file_dir):
    #     if doc_id in filename:
    #         os.remove(filename) 

    print(output)
    return output

##
# print(torch.cuda.is_available())
biomedical_evet_extraction("The B cells were found to express BMP type I and type II receptors and BMP-6 rapidly induced phosphorylation of Smad1/5/8.")