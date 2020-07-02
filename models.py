from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from transformers import *
import itertools
from utils import *
from optimization import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm 
from utils import timer
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import copy
from transformers.modeling_bert import BertIntermediate, BertOutput, BertLayer, BertSelfOutput, BertLayer
from collections import defaultdict
try:
    from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, NNConv, GCNConv, GraphUNet, GATConv
except:
    pass

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

class EdgeNet(nn.Module):
    '''
    A neural network ℎ𝚯 that maps edge features edge_attr of shape [-1, num_edge_features] to shape [-1, in_channels * out_channels], e.g., defined by torch.nn.Sequential.
    '''
    def __init__(self, num_edge_embeddings, edge_embedding_dim, output_dim, kg_pretrained_weights=None):
        super(EdgeNet, self).__init__()
        # self.linear1 = nn.Linear(num_edge_features, num_edge_features)
        
        
        if kg_pretrained_weights is not None:
            edge_embedding_weights = torch.load(kg_pretrained_weights)['rel_embeddings.weight']
            self.edge_embedding = nn.Embedding.from_pretrained(edge_embedding_weights, freeze=False)
        else:
            self.edge_embedding = nn.Embedding(num_embeddings=num_edge_embeddings, embedding_dim=edge_embedding_dim)

        self.linear2 = nn.Linear(edge_embedding_dim, output_dim)
        
    def forward(self, edge_attr):
        
        # sum of embedding
        embedding = torch.matmul( edge_attr, self.edge_embedding.weight)
        
        x = F.relu(self.linear2(embedding))
        
        
        return x



class KnowledgeGNN(nn.Module):

    
    def __init__(self, kg_embedding_dim, num_edge_embeddings, token_embedding_size, args, kg_pretrained_weights=None):
        super(KnowledgeGNN, self).__init__()

        class ECGAT(NNConv):
            '''
            Edge Conditioned Graph Attention Network
            '''
            def __init__(self, in_channels, out_channels, nn):
                super(ECGAT, self).__init__(in_channels, out_channels, nn)
                self.att = torch.nn.Linear(2 * out_channels, 1)
                
                self.negative_slope = 0.2
                self.reset_parameters()

            
            def message(self, edge_index_i, x_i, x_j, pseudo):
                '''
                pseudo: edge_attr.unsqueeze(-1)
                x_j: neighboring node embeddings
                '''
                
                weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
                neighbors = torch.matmul(x_j.unsqueeze(1), weight).squeeze(dim=1)
                
                alpha = torch.cat([x_i, neighbors], dim=1) 
                
                alpha = self.att(alpha).unsqueeze(dim=1)
                # alpha = F.leaky_relu(alpha, self.negative_slope)
                
                alpha = F.softmax(alpha , dim=0)
                
                return x_j * alpha.view(-1, 1)

            def update(self, aggr_out, x):
                
                if self.root is not None:
                    aggr_out = aggr_out + torch.mm(x, self.root)
                    
                if self.bias is not None:
                    aggr_out = aggr_out + self.bias
                return aggr_out

        

        self.kg_embedding_dim = kg_embedding_dim
        self.edge_mlp = EdgeNet(num_edge_embeddings, kg_embedding_dim, kg_embedding_dim* kg_embedding_dim)

        
        if args.gnn_type == 'ECGAT':
            self.conv1 = ECGAT(kg_embedding_dim, kg_embedding_dim, self.edge_mlp)
            self.conv2 = ECGAT(kg_embedding_dim, kg_embedding_dim, self.edge_mlp)
        if args.gnn_type == 'NNConv':
            self.conv1 = NNConv(kg_embedding_dim, kg_embedding_dim, self.edge_mlp)
            self.conv2 = NNConv(kg_embedding_dim, kg_embedding_dim, self.edge_mlp)
        
        self.dropout1 = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(token_embedding_size, kg_embedding_dim)
        self.linear2 = nn.Linear(kg_embedding_dim, token_embedding_size)
        self.nt_linear = nn.Linear(kg_embedding_dim, 3)
        self.edge_linear = nn.Linear(kg_embedding_dim * 2, num_edge_embeddings)
        if kg_pretrained_weights is not None:
            kg_embedding_weights = torch.load(kg_pretrained_weights)['ent_embeddings.weight']
            self.kg_embedding = nn.Embedding.from_pretrained(kg_embedding_weights, freeze=False)
        else:
            self.kg_embedding = nn.Embedding(num_embeddings=args.num_node_embeddings, embedding_dim=kg_embedding_dim)
        self.mnc_linear = nn.Linear(kg_embedding_dim, args.num_node_embeddings)
        
    def forward(self, node_ids, edge_index, edge_attr, token_embeddings, node_type_labels, num_recognized_tokens, mask_out_rate, args):
        
        '''
        edge_index: [2, num_edges]
        edge_attr: [num_edges, num_edge_features]
        '''
        edge_attr = edge_attr.float()

        token_embeddings = self.linear1(token_embeddings)

        node_embeddings = self.kg_embedding(node_ids)

        # token, CUI, STY
        all_node_embeddings = torch.cat([token_embeddings, node_embeddings], dim=0)
        
        
        
        # if there is no dropped nodes, masked_node equal to all nodes
        masked_node_embeddings = all_node_embeddings

        # only mask out node during training
        if self.training and args.mnc:
            
            # create mask for nodes
            masked = torch.rand(len(node_ids))
            node_mask = (masked > mask_out_rate)

            if (~node_mask).sum() > 0:
                node_mask = node_mask.cuda()

                # the index of the node to be drop
                drop_nodes = torch.where(~ node_mask)

                node_labels = node_ids[drop_nodes]  

                # keep all token nodes
                token_node_mask = torch.ones([token_embeddings.size(0)]).bool().cuda()
                node_mask = torch.cat([token_node_mask, node_mask], axis=0)

                # the index of the node to be drop on the whole graph
                drop_nodes = torch.where(~ node_mask)

                node_mask = torch.cat([node_mask.unsqueeze(1)] * self.kg_embedding_dim, dim=1)

                masked_node_embeddings = all_node_embeddings * node_mask
            
        
            

        x1 = self.conv1(masked_node_embeddings, edge_index, edge_attr)
        x = F.relu(x1)
        x = self.dropout1(x)


        x2 = self.conv2(x, edge_index, edge_attr)
        
        # x = F.relu(x2)
        # x = self.dropout1(x)

        # x3 = self.conv3(x, edge_index, edge_attr)
        final_x = x2

        
        # KGE loss 
        edge_embeddings = torch.matmul( edge_attr, self.edge_mlp.edge_embedding.weight)
        # # edge_embeddings /= edge_attr.sum(dim=1).unsqueeze(1)

        # # projection_matrices = torch.matmul(edge_attr, self.projection_matrix.weight)
        kge_loss = 0
        # # kge_loss_fn = BCELoss()
        kge_loss_fn = MSELoss()
        
        
        # edge classification loss
        if args.edge_cls:
            kge_loss_fn = BCELoss() 
            labels = edge_attr
            
            head_embeddings = final_x[edge_index[0,:]]
            tail_embeddings = final_x[edge_index[1,:]]
            head_tail = torch.cat([head_embeddings, tail_embeddings], dim=1)
            predicted = torch.nn.Sigmoid()(self.edge_linear(head_tail))
            
            kge_loss = 0
            
            kge_loss = kge_loss_fn(predicted, labels)
        
        # link prediction loss
        if args.link_pred:

            kge_loss_fn = MSELoss(reduction='none')

            
            predicted_tails = final_x[edge_index[0,:]] + edge_embeddings
            edge_labels = final_x[edge_index[1,:]]

            # # compute loss on those those not are tok to CUI or CUI to STY
            mask = edge_attr[:,-3:].sum(dim=1) == 0
            mask = torch.cat([mask.unsqueeze(1)] * edge_labels.size(1), dim=1)
            
            kge_loss = kge_loss_fn(predicted_tails, edge_labels)
            
            kge_loss = (kge_loss * mask).mean()
        
        # node type classification loss
        nt_loss = 0
        if args.nt_cls:
            nt_pred = self.nt_linear(final_x)
            nt_loss_fn = CrossEntropyLoss()
            nt_loss += nt_loss_fn(nt_pred, node_type_labels)

        # masked node classificatoin loss
        mnc_loss = 0
        if self.training and args.mnc and (~node_mask).sum() > 0:
            
            # mce loss
            prediction = self.mnc_linear(final_x[drop_nodes])
            
            # compute loss
            mnc_loss_fn = CrossEntropyLoss()

            mnc_loss += mnc_loss_fn( prediction, node_labels)
        

        # compute final outputs
        final_outputs = self.linear2(final_x)

        return final_outputs, kge_loss, nt_loss, mnc_loss
        

class MultitaskClassifierBase(nn.Module):

    def __init__(self):
        super(MultitaskClassifierBase, self).__init__()     

    def forward(self, input_ids, entity_labels,
                attention_mask=None, token_type_ids=None, kg_datas=None, position_ids=None, head_mask=None, rel_idxs=[], lidx=[], ridx=[], task='relation', args=None):
        '''
        entity_labels are just for extracting proteins
        '''
        
        
        out = self.bert(input_ids, 
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask)
        out = self.dropout(out[0])
        if args.use_knowledge:
            predicted_edges = defaultdict(list)
            if len(rel_idxs) > 0:
                for b,r in rel_idxs:
                    predicted_edges[b].append((lidx[b][r], ridx[b][r]))
                    predicted_edges[b].append((ridx[b][r], lidx[b][r]))
        
        kge_loss = 0
        node_type_losses = 0
        mnc_losses = 0
        if args.use_knowledge:
            for batch, kb_feature in enumerate(kg_datas):            

                if len(kb_feature['nodes']) > 0:
                    # place tensor to GPU 
                    kb_feature = {k: v.cuda() if k not in ['CUI_length','num_recognized_tokens','tokens'] else v for k,v in kb_feature.items() }
                    nodes, edge_index, edge_attr = kb_feature['nodes'], kb_feature['edge_index'], kb_feature['edge_attr']
                    CUI_length = kb_feature['CUI_length']
                    edge_attr = edge_attr.float()
                    
                    num_recognized_tokens = kb_feature['num_recognized_tokens']

                    # tokens
                    token_ids = nodes[:num_recognized_tokens]

                    # construct predicted edges
                    if args.use_temporal_edge:
                        # add an empty column for the initial attr without temproal  dimension
                        empty_attr = torch.zeros([edge_attr.size(0), 1], dtype=torch.float).cuda()
                        edge_attr = torch.cat([edge_attr, empty_attr], dim=1)
                        sent_token_idx2graph_token_idx = { sent_token_idx.item(): graph_token_idx for graph_token_idx, sent_token_idx in enumerate(token_ids)}
                        token_relation_edge_index = [ [sent_token_idx2graph_token_idx[l], sent_token_idx2graph_token_idx[r]]  for l, r in predicted_edges[batch] if l in sent_token_idx2graph_token_idx and r in sent_token_idx2graph_token_idx]
                        
                        if len(token_relation_edge_index) > 0:
                            # [2, edges]
                            token_relation_edge_index = torch.tensor(token_relation_edge_index).T.cuda()
                            edge_index = torch.cat([edge_index, token_relation_edge_index], dim=1)
                            
                            # num_edge, edge_embedding_size
                            temporal_edge_attr = torch.zeros([ token_relation_edge_index.size(1) , args.num_edge_embeddings], dtype=torch.float).cuda()
                            
                            # the temporal attribute is at the last dimension
                            temporal_edge_attr[:,-1] = 1
                            edge_attr = torch.cat([edge_attr, temporal_edge_attr], dim=0)
                            # print(temporal_edge_attr.size())
                    
                    
                    # CUI & STY
                    node_ids = nodes[num_recognized_tokens:]

                    node_type_labels = [0] * num_recognized_tokens + [1] * CUI_length + [2] * (len(nodes) - num_recognized_tokens - CUI_length)
                    node_type_labels = torch.tensor(node_type_labels, dtype=torch.long).cuda()

                    token_embeddings = out[batch, token_ids, :]

                    
                    node_embeddings, loss, node_type_loss, mnc_loss = self.gnn(node_ids, edge_index, edge_attr, token_embeddings, node_type_labels, num_recognized_tokens, mask_out_rate=0.2, args=args)
                    
                    
                    
                    # get embedding only for token nodes
                    token_embeddings = node_embeddings[:len(token_ids), :]
                    # token_embeddings = node_embeddings[len(token_ids):, :]

                    kge_loss += loss
                    node_type_losses += node_type_loss
                    mnc_losses += mnc_loss
                    # outputs[batch, token_ids,:] = token_embeddings
                    out[batch, token_ids,:] += token_embeddings
                    
                    # out[batch,:,:] += attended_outs
                    
            
        

        ### entity prediction - predict each input token
        if task == 'entity':
            
            # gather protein embeddings
            protein_mask =  torch.cat( out.size(2) * [(entity_labels==9).unsqueeze(2)], dim=2)
            protein_embeddings = out * protein_mask
            
            protein_embeddings = F.avg_pool1d(protein_embeddings.permute(0,2,1), kernel_size=protein_embeddings.size(1), count_include_pad=False).permute(0,2,1)
            
            protein_embeddings = torch.cat( out.size(1) * [protein_embeddings], dim=1)

            # concatenate protein embeddings with the whole sequence
            out = torch.cat([out, protein_embeddings], dim=2)

            out_ent = self.linear1_ent(out)
            out_ent = self.act(out_ent)
            out_ent = self.linear2_ent(out_ent)
            prob_ent = self.softmax_ent(out_ent)

            # mask = torch.cat([attention_mask] * prob_ent.size(2), dim=2)
            
            return out_ent, prob_ent, kge_loss, node_type_losses, mnc_losses

        ### relaiton prediction - flatten hidden vars into a long vector
        if task == 'relation':
            
            
            # out : [2, 114, 200]
            ltar_f = torch.cat([out[b, lidx[b][r], :].unsqueeze(0) for b,r in rel_idxs], dim=0)
            
            rtar_f = torch.cat([out[b, ridx[b][r], :].unsqueeze(0) for b,r in rel_idxs], dim=0)
            # rtar_b = torch.cat([out[b, ridx[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)

            # out: [12, 401]
            out = self.dropout(torch.cat((ltar_f, rtar_f), dim=1))
            # out = torch.cat((out, fts), dim=1)

            # linear prediction
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.linear2(out)
            prob = self.softmax(out)
            return out, prob, 0, 0, 0


    def construct_relations(self, entity_logits, entity_labels, attention_masks, interactions, interaction_labels, args, gold=True, test=False):
        '''
        ent_probs: Predicted entity probabilty [batch, seq, n_ent_class]
        ents: Gold entities [batch, seq, ], just for identiying proteins
        lengths: The length of each packed sequence [batch]
        pairs: golden 
        ints: interactions
        doc: list of sentence_id
        poss: pos tags
        '''

        
        nopred_rels = []

        ## Case 1: only use gold relation
        if gold:
            # pred_rels = rels
            pred_rels = interactions

        ## Case 2: use candidate relation predicted by entity model
        else:
            
            def _is_gold(pair_pred, pairs_gold):
                return pair_pred in pairs_gold

            batch_size = entity_logits.size(0)
            
            

            
            # ent_preds = ent_probs.max(dim=2, keepdim=False)[1].tolist()
            predicted_entities = entity_logits.argmax(dim=2)
            
            # protein_mask
            protein_id = args._label_to_id_t['Protein']  
            none_entity_id = args._label_to_id_t['Protein']

            pred_ints = []
            pred_pairs = []
            for i in range(len(predicted_entities)):

                predicted_entity = predicted_entities[i]
                entity_label = entity_labels[i]
                attention_mask = attention_masks[i]

                # if test, then don't get interaction label 
                if not test:
                    interaction = interactions[i]
                    interaction_label = interaction_labels[i]

                predicted_entity = predicted_entity.cpu().numpy()
                # get the position of all gold proteins in the sentences
                gold_prot_idxs = np.where(entity_label.cpu() == protein_id)[0]
                
                
                # get the position of all predicted triggers in the sentences
                tri_idxs = np.where((predicted_entity > 0) *( predicted_entity != 9))[0].tolist()

                
                # trigger entity pairs
                te_pairs = list(itertools.product(tri_idxs, gold_prot_idxs))
                
                
                # trigger trigger pairs
                tt_pairs = [(i, j) for i in tri_idxs for j in tri_idxs if i != j and args._id_to_label_t[predicted_entity[i]] in args.REG]

                
                pred_int = []
                pred_pairs.append(te_pairs+tt_pairs)
                
                if not test:
                    
                    for p in te_pairs+tt_pairs:
                        
                        if _is_gold(p, interaction):
                            
                            pred_int.append(interaction_label[interaction.index(p)])
                        else:
                            # None event
                            pred_int.append(args._label_to_id_i['None'])
                    pred_ints.append(pred_int)

            
            
            pred_pairs = tuple(pred_pairs)
            pred_ints = tuple(pred_ints)

            pred_rels = pred_pairs
            interaction_labels = pred_ints

        rel_idxs, lidx, ridx = [],[],[]
        
        for i, rel in enumerate(pred_rels):
            rel_idxs.extend([(i, ii) for ii, _ in enumerate(rel)])
            lidx.append([x[0] for x in rel])
            ridx.append([x[1] for x in rel])

        # if test, don't return relation label
        if test:
            return None, rel_idxs, lidx, ridx


        rels = [x for rel in pred_rels for x in rel]
        if rels == []:
            labels = torch.FloatTensor([])
        else:
            labels = torch.LongTensor([x for rel in interaction_labels for x in rel])

        # pdb.set_trace()
        
        return labels, rel_idxs, lidx, ridx


    def predict(self, dev_dataloader, gold, args, dev_interactions=None, dev_interaction_labels=None, test=False, eval_kg_datas=None):
        
        self.eval()
        # need to have a warm-start otherwise there could be no event_pred
        # may need to manually pick poch < #, but 0 generally works when ew is large
        
        with torch.no_grad():
            predicted_interactions = []
            predicted_interaction_labels = []
            predicted_entities = []
            all_gold_interactions = []
            all_gold_interaction_labels = []
            all_gold_entities = []
            all_input_ids = []
            all_sample_ids = []

            ent_pred_map, ent_label_map = {}, {}
            rd_pred_map, rd_label_map = {}, {}
            
            y_trues_e, y_preds_e = [], []
            y_trues_r, y_preds_r = [], []
            for step, batch in enumerate(tqdm(dev_dataloader, desc='Prediction')):
            
                if torch.cuda.is_available():
                    # put the variables onto GPU
                    batch = tuple(t.cuda() for t in batch)
                
                dev_input_ids, dev_input_masks, dev_segment_ids, dev_entity_labels, dev_sample_ids = batch
                
                
                # get sample ids only for data_out
                all_sample_ids.extend(dev_sample_ids.cpu().numpy())
                
                if args.use_knowledge:
                    kg_datas = [ eval_kg_datas[sample_id] for sample_id in dev_sample_ids]
                else:
                    kg_datas = None
                    
                # entity output
                entity_logits, prob_e, _, _, _ = self.forward(dev_input_ids, dev_entity_labels, dev_input_masks, dev_segment_ids, kg_datas=kg_datas, task='entity',args=args)   # out_e and prob_e: [16, 56, 11]

            
                # mask out the prob of the padding with input mask        
                mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (entity_logits.size(2) ),dim=2)
                mask[:,:,0] = 1
                prob_e *= mask
                
                

                
                if not test:
                    gold_interactions = [dev_interactions[sample_id] for sample_id in dev_sample_ids]
                    gold_interaction_labels = [dev_interaction_labels[sample_id] for sample_id in dev_sample_ids]
                    
                    all_gold_interactions.extend(gold_interactions)
                    all_gold_interaction_labels.extend([args._id_to_label_i[label] for labels in gold_interaction_labels for label in labels])
                    # construct relation
                    label_r, rel_idxs, lidx, ridx = self.construct_relations(prob_e, dev_entity_labels, dev_input_masks, gold_interactions, gold_interaction_labels, args, gold=gold, test=test)
                else:
                    label_r, rel_idxs, lidx, ridx = self.construct_relations(prob_e, dev_entity_labels, dev_input_masks, None, None, args, gold=gold, test=test)
                
                assert len(lidx) == len(ridx)

                # retrieve the predicted pairs
                pair_lengths = [len(i) for i in lidx]  # num of pairs in each sent in the batch
                for i in range(len(lidx)): # batch size
                    if len(lidx[i]) == 0:
                        predicted_interactions.append([])
                    else:
                        predicted_interactions.append([i for i in zip(lidx[i], ridx[i])])
                
                ### predict relations
                if rel_idxs != []: # predicted relation could be empty --> skip

                    
                    out_r, prob_r, _, _, _ = self.forward(dev_input_ids, dev_input_masks, dev_segment_ids, kg_datas=kg_datas, rel_idxs=rel_idxs, lidx=lidx, ridx=ridx, task='relation', args=args)
                    
                    # (batch, )
                    pred_r = prob_r.data.argmax(dim=1).long().view(-1)
                    if not test:
                        assert pred_r.size(0) == label_r.size(0)

                    if args.cuda:
                        prob_r = prob_r.cpu()
                        if not test:
                            label_r = label_r.cpu()
                    
                    pred_r_list = pred_r.tolist()
                    # extend to all predicted relations
                    y_preds_r.extend(pred_r_list)

                    # retrive the ints labels for the predicted pairs
                    cur = 0
                    for i, l in enumerate(pair_lengths):
                        if pair_lengths[i] == 0:
                            predicted_interaction_labels.append([])
                        else:
                            predicted_interaction_labels.append([args._id_to_label_i[x] for x in pred_r_list[cur:cur+l]])
                            cur += l

                else: # no relation predicted

                    y_preds_r.extend([])
                    predicted_interaction_labels.extend([[] for _ in range(len(dev_input_masks))])

                assert len(predicted_interaction_labels[-1]) ==len(predicted_interactions[-1])
                    
                if not test:
                    y_trues_r.extend(label_r.tolist())

                # retrieve and flatten entity prediction for loss calculation
                ent_pred, ent_label, ent_prob, ent_key, ent_pos, ent_input = [], [], [], [], [], []

                # get entities prediction filtered by mask
                for i, mask in enumerate(dev_input_masks):
                    
                    mask = mask.bool()
                    # take only mask==1 portion
                    ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
                    
                    # flatten entity label
                    ent_label.append(torch.masked_select(dev_entity_labels[i], mask))
                    
                    ent_input.append(torch.masked_select(dev_input_ids[i], mask))

                    all_gold_entities.append(ent_label[-1].tolist())
                    predicted_entities.append(ent_pred[-1].tolist())
                    all_input_ids.append(ent_input[-1].tolist())
                    
                ## collect trigger prediction results
                ent_pred = torch.cat(ent_pred, 0)
                ent_label = torch.cat(ent_label, 0)
                
                
                
                assert ent_pred.size() == ent_label.size() 


                y_trues_e.extend(ent_label.tolist())
                y_preds_e.extend(ent_pred.tolist())

                
                    
                
                    
                

            data_out = {'sample_ids':all_sample_ids, 
            'predicted_entities':predicted_entities,  
            'predicted_interactions': predicted_interactions, 
            'predicted_interaction_labels':predicted_interaction_labels,
            'gold_entities': all_gold_entities,
            'gold_interactions': all_gold_interactions,
            'gold_interaction_labels': all_gold_interaction_labels,
            'input_ids':all_input_ids,
            
            

            }

        return y_trues_e, y_preds_e, y_trues_r, y_preds_r, data_out
    
    

class BertMultitaskClassifier(MultitaskClassifierBase):
    
    def __init__(self, args, bert_weights_path='biobert_weights/scibert_scivocab_uncased', kg_pretrained_weights=None):
        super(BertMultitaskClassifier, self).__init__()     
        

        

        self.bert = BertModel.from_pretrained(bert_weights_path)
        config = self.bert.config
        kg_embedding_dim = args.kg_embedding_dim
        self.config = config
        self.hid_size = args.hid
        
        
        self.num_classes = max(args._label_to_id_i.values()) + 1
        self.num_ent_classes = max(args._label_to_id_t.values()) + 1

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(config.hidden_size *2, args.rel_linear_size)
        self.linear2 = nn.Linear(args.rel_linear_size, self.num_classes)

        # MLP classifier for entity
        self.linear1_ent = nn.Linear(config.hidden_size * 2 , args.ent_linear_size)
        self.linear2_ent = nn.Linear(args.ent_linear_size, self.num_ent_classes)

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_ent = nn.Softmax(dim=2)
        if args.use_knowledge:
            self.gnn = KnowledgeGNN(kg_embedding_dim=kg_embedding_dim, num_edge_embeddings=args.num_edge_embeddings, token_embedding_size=config.hidden_size, args=args, kg_pretrained_weights=kg_pretrained_weights)
        