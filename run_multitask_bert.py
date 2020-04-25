from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from dataclasses import dataclass
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
from utils import timer
from models import BertMultitaskClassifier
import pdb
from optimization import *
import logging
# torch.autograd.set_detect_anomaly(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)

# logger = logging.getLogger(__name__)
def main(args):

    data_dir = args.data_dir

    
    data_test = pickle.load(open(os.path.join(args.data_dir, args.test_pkl), 'rb'))
    
    test_kg_datas = None
    if args.use_knowledge:
        test_kg_datas = pickle.load(open(f'{args.data_dir}/{args.test_kg_datas}.pkl','rb'))
    
     

    # args._label_to_id_t = OrderedDict([('None', 0), ('Gene_expression', 1), ('Localization', 2), ('Transcription', 3), ('Binding', 4), ('Phosphorylation', 5), ('Positive_regulation', 6), ('Regulation', 7), ('Protein_catabolism', 8), ('Protein', 9), ('Negative_regulation', 10), ('Entity', 0)])
    # args._id_to_label_t = {0: 'None', 1: 'Gene_expression', 2: 'Localization', 3: 'Transcription', 4: 'Binding', 5: 'Phosphorylation', 6: 'Positive_regulation', 7: 'Regulation', 8: 'Protein_catabolism', 9: 'Protein', 10: 'Negative_regulation'}
    # args._label_to_id_i = OrderedDict([('None', 0), ('Theme', 1), ('Cause', 2), ('Site', 0), ('ToLoc', 0), ('AtLoc', 0), ('SiteParent', 0)])
    # args._id_to_label_i = {0: 'None', 1: 'Theme', 2: 'Cause'}

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
        
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        print(output_model_file)
        checkpoint = torch.load(output_model_file)
        model.load_state_dict(checkpoint)

        model.cuda()
        model.eval()
        gold=False
        y_trues_e, y_preds_e, y_trues_r, y_preds_r, data_out = model.predict(test_dataloader, gold, args, test=True, eval_kg_datas=test_kg_datas)
         


        write_pkl(data_out, test_abs_spans, test_doc_ids, args.out_test_pkl, gold_tri=False)
        unmerge_normalize(args.output_dir.split('/')[-1])

    
def unmerge_normalize(model_name):
    '''
    Call unmerge & normalize on test set
    '''

    output_dir='genia_cord_19_output'
    unmerge_cmd=f'python unmerg_write.py -pred_pkl={args.out_test_pkl}'\
    ' -protIdBySpan=preprocessed_data/CORD_19_PMCprotIdBySpan.pkl'\
    ' -origIdById=preprocessed_data/CORD_19_PMCorigIdById.pkl'\
    f' -out_dir={output_dir}'
    p = subprocess.Popen(unmerge_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    
    # normalization
    normalize_cmd = f'./eval/tools/a2-normalize.pl -g genia_cord_19/ -u {output_dir}/*.a2'
    p = subprocess.Popen(normalize_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    # copy a1 & txt files to a2
    [ shutil.copy(source_file, output_dir) for file in os.listdir(output_dir) for source_file in [f'genia_cord_19/{file.split(".a2")[0]}.txt', f'genia_cord_19/{file.split(".a2")[0]}.a1']]
        
            
    

def write_pkl(data_out, abs_spans, doc_ids, out_file, gold_tri=True):
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
    print('out pickle file saved as {}'.format(out_file))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str, default = './preprocessed_data/')
    p.add_argument('-other_dir', type=str, default = '../other')
    # select model
    p.add_argument('-model', type=str, default='multitask/pipeline')#, 'multitask/gold', 'multitask/pipeline'
    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=100)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument("-train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    p.add_argument("-eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")                    
    # p.add_argument('-data_type', type=str, default="matres")
    p.add_argument('-epochs', type=int, default=30)
    p.add_argument('-pipe_epoch', type=int, default=1000) # 1000: no pipeline training; otherwise <= epochs
    # p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=5e-5)
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.2)
    p.add_argument('-ngbrs', type=int, default = 15)
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-w2i', type=OrderedDict)
    p.add_argument('-glove', type=OrderedDict)
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-uw', type=float, default=1.0)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=1)
    p.add_argument('-relation_weight', type=float, default=1.0)
    p.add_argument('-entity_weight', type=float, default=1.0)
    p.add_argument('-save_model', type=bool, default=True)
    p.add_argument('-save_stamp', type=str, default="matres_entity_best")
    p.add_argument('-entity_model_file', type=str, default="singletask_trigger_EXPtest_TF10.7015_RF10.3919_WF10.7015.pth.tar")
    p.add_argument('-relation_model_file', type=str, default="singletask_relation_EXPtest_TF10.0048_RF10.8791_WF10.8791.pth.tar")
    p.add_argument('-entity_model_file_e2e', type=str, default="singletask_end2end_trigger_EXPtest_TF10.7020_RF10.8633_WF10.8296.pth.tar")
    p.add_argument('-relation_model_file_e2e', type=str, default="singletask_end2end_relation_EXPtest_TF10.7020_RF10.8633_WF10.8296.pth.tar")
    p.add_argument('-load_model', type=str2bool, default=False)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-fine_tune', type=bool, default=False)
    p.add_argument('-eval_gold',type=bool, default=True)
    # new_add
    p.add_argument('-use_pos', type=str2bool, default=True)
    p.add_argument('-trainable_emb', type=str2bool, default=False)
    p.add_argument('-opt', choices=['adam', 'sgd', 'adagrad'], default='adagrad')
    p.add_argument('-exp_id', type=str, default='test')
    p.add_argument('-random_seed', type=int, default=123)
    p.add_argument('-output_dir', type=str, default='local_models/')
    p.add_argument('-nw_r', type=float, default=1.0, help='weight for None in relation')
    p.add_argument('-nw_e', type=float, default=1.0, help='weight for None in trigger')
    p.add_argument('-regen_vocfile', type=str2bool, default=False)
    p.add_argument('-do_eval', type=str2bool, default=False, help='load trained models and predict to write pkls, \
                                                                                Note this is different from args.load_model. This is used only when you have the trained models \
                                                                                and want to write out the pkl')
    p.add_argument('-do_train', type=str2bool, default=False, help='true to train the model')                                                                                
    p.add_argument('-do_test', type=str2bool, default=False, help='true to predict on the test set')                                                                                    
    p.add_argument('-test_on_gold_tri', type=str2bool, default=True)
    p.add_argument('-tune_t', type=str2bool, default=True, help='used with load_model, decide wether the loaded trigger model will be updated during end2end')
    
    p.add_argument('-out_test_pkl', type=str, default='CORD_out_merged.pkl')
    
    p.add_argument('-test_pkl', type=str, default='CORD_19_PMC.pkl')

    p.add_argument('-train_kg_datas', type=str)
    p.add_argument('-dev_kg_datas', type=str)
    p.add_argument('-test_kg_datas', type=str)
    p.add_argument('-kg_pretrained_weights', type=str)
    p.add_argument('-kg_embedding_dim', type=int)
    p.add_argument('-ent_linear_size', type=int)
    p.add_argument('-rel_linear_size', type=int)
    p.add_argument('-use_knowledge', type=str2bool, default=True)
    p.add_argument('-use_temporal_edge', type=str2bool, default=False)
    p.add_argument('-link_pred', type=str2bool, required=True, help="Enable link prediction loss")
    p.add_argument('-nt_cls', type=str2bool, required=True, help="Enable node type classification loss")
    p.add_argument('-edge_cls', type=str2bool, default=False, help="Enable edge classification loss")
    p.add_argument('-mnc', type=str2bool, required=True, help="Enable masked node classification loss")
    p.add_argument('-gpu', type=str, required=True)
    p.add_argument('-gnn_type', type=str)
    p.add_argument("-local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    p.add_argument('-gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    p.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    args = p.parse_args()
    args.save_stamp = "%s_hid%s_dropout%s_ew%s" % (args.save_stamp, args.hid, args.dropout, args.entity_weight)

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

    print(f"Output Directory {args.output_dir}")

    
    main(args)

