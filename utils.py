import time
from contextlib import contextmanager
import json
import os
from glob import glob
import re
import numpy as np
import spacy
from scispacy.umls_linking import UmlsEntityLinker
from transformers import *
import pandas as pd
import pickle
from collections import defaultdict


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.5f} s'.format(name,(time.time() - t0)))


NER_tagger = spacy.load('en_ner_jnlpba_md')
# linker = UmlsEntityLinker(resolve_abbreviations=True)
# NER_tagger.add_pipe(linker)


# load bert tokenizer
bert_dir = 'allenai/scibert_scivocab_uncased'
# bert_dir = '../GE_event/bert_weights/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(bert_dir)

# df that map CUI to STY
# with open('umls_data/MRSTY.RRF','r') as f:
#     lines = [line.split('|')[:-1] for line in f.readlines()]

# mapping_df = pd.DataFrame(lines,columns=['CUI','TUI','STN','STY','ATUI','CVF'])
# mapping_df.head()

# # filter out the df such that it only contains protein CUI
# mapping_df = mapping_df.loc[mapping_df.STY=='Amino Acid, Peptide, or Protein']
# protein_CUIs = set(mapping_df.CUI)

max_length=230
dummy_span = None

def filter_protein_entities(entities, document):
    '''
    Feed a list of entities, and return the span of those whose type is proein
    '''
    # store the start and end of each mapped entity
    document_entities = []

    for entity in entities:
        
        # if the entity is a protein
        if document[entity.start].ent_type_ == 'PROTEIN':
            document_entities.append(entity)
        
    return document_entities

def generate_sentence_instance(document, sentence, doc_id, ent_char2_id_map, corpus_proteinOrigIdBySpan):
    '''
    Generate one instance per sentence.
    '''
    
    # only mark the head tokens as label

    entities = sentence.ents
    
    # start char offset of this sentence    
    sent_start = sentence.start_char
    entities = filter_protein_entities(entities, document)
    # don't consider sentence with no entity 
    # if len(entities) == 0: return None
    
    # obtain the text of this sentence
    text = str(sentence)
    
    # head entity char position -> bert head token position
    # TODO: create origin to token map
    char_to_token_map = {}
    token_to_char_map = {}
    
    char_to_token_tail_map = {}
    new_tokens = ["[CLS]"]
            
    # cumulative the cumulative sum of length in origin_tokens except for [CLS]
    cur_tokens_length = 0
    

    # split sentence into sub-sentences
    if len(entities) > 0:
        segment_numbers = [0]+\
        np.hstack([[entity.start_char - sent_start, entity.end_char - sent_start]  for entity in entities ]).tolist()\
        + [len(text)+1]   
    else:
        segment_numbers = [0]
    
    sub_sentences = [text[start:end] for start, end in zip(segment_numbers[:-1], segment_numbers[1:])]

    # absolute span of each entity
    abs_spans = [dummy_span]

    for sub_sentence in sub_sentences:
        # create a mapping from character space to bert token space

        skip_patterns = r'([^a-zA-Z0-9])'

        for original_texts in re.split(skip_patterns, sub_sentence):
        
            if len(original_texts) == 0: continue

            # a list of bert tokens
            bert_tokens = tokenizer.tokenize(original_texts)
            
            head_char_position = cur_tokens_length 

            # head token position
            char_to_token_map[head_char_position] = len(new_tokens)
            token_to_char_map[len(new_tokens)] = head_char_position

            cur_tokens_length += len(original_texts)
            new_tokens.extend(bert_tokens)
            
            tail_char_position = cur_tokens_length -1 

            # if the tail_char_position is a list
            while text[tail_char_position] in [' '] and tail_char_position >= head_char_position:
                tail_char_position -= 1
            
            # this is invalid
            if tail_char_position < head_char_position or re.search(skip_patterns, original_texts) is not None: #original_texts=='\n':# 

                abs_spans.extend([dummy_span] * (len(bert_tokens) ))                
                continue
            
            tail_position = len(new_tokens) 
            
            # trim the tail token position if the token is not [UNK]
            if new_tokens[tail_position-1] not in ['[UNK]']:
                while text[tail_char_position].lower() != new_tokens[tail_position-1][-1]:
                    # tail token position
                    tail_position -= 1


            char_to_token_tail_map[tail_char_position] = len(new_tokens) 
            abs_span = '-'.join([str(head_char_position+sent_start), str(tail_char_position+1+sent_start)])                    

            abs_spans.append(abs_span)  

            # append dummy spans
            abs_spans.extend([dummy_span] * (len(bert_tokens) -1))
    
    entity_label = np.array(['None'] * (len(new_tokens) + 1) ,dtype=object)  # plus 1 for [SEP]

    # iteratev over each offset
    for entity in entities:

        head_char_idx = entity.start_char - sent_start
        tail_char_idx = entity.end_char - sent_start

        # only add if there is a None
        assert entity_label[char_to_token_map[head_char_idx]] == 'None'
        
        # assign protein label
        entity_label[char_to_token_map[head_char_idx]] = 'Protein'



        # doc_id -> span -> protein origid
        corpus_proteinOrigIdBySpan[doc_id][abs_spans[char_to_token_map[head_char_idx]]] = f'{doc_id}.{ent_char2_id_map[entity.start_char]}'



        # make sure at least the first character of the mapping is correct
        assert new_tokens[char_to_token_map[head_char_idx]][0] == text[head_char_idx].lower(), (new_tokens[char_to_token_map[head_char_idx]][0] , text[head_char_idx], entity)


    # convert entity label back to list for zero padding
    entity_label = entity_label.tolist()


    new_tokens.append('[SEP]')
    abs_spans.append(dummy_span)            

    assert len(new_tokens) == len(abs_spans), (len(new_tokens) , len(abs_spans))


    if len(new_tokens) > max_length:
        print(f"{doc_id}-{sentence.ent_id}: Exceed max length")
        return None
    tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    # mask ids
    mask_ids = [1] * len(tokenized_ids)

    # segment ids
    segment_ids = [0] * len(tokenized_ids)

    if len(tokenized_ids) < max_length:
        # Zero-pad up to the sequence length
        padding = [0] * (max_length - len(tokenized_ids))
        tokenized_ids += padding
        entity_label += ['None'] * len(padding)
        mask_ids += padding
        segment_ids += padding

    assert len(tokenized_ids) == max_length == len(mask_ids) == len(segment_ids) == len(entity_label ) ,\
     (len(tokenized_ids) ,max_length ,len(mask_ids) ,len(segment_ids), len(entity_label) )


    result_dict = {
    'tokenized_ids': tokenized_ids,
    'entity_labels': entity_label,
    'mask_ids': mask_ids,
    'segment_ids': segment_ids,
    'sent_ids':sentence.ent_id,
    'doc_ids': doc_id,
    'token_to_char_map': token_to_char_map,
    'char_to_token_map': char_to_token_map,
    'abs_spans':abs_spans
    }
    return result_dict



def process_document(document, doc_id, output_dir, corpus_proteinOrigIdBySpan):
    '''
    Run Spacy to extract proteins and split single document into sentences
    returns:
        doc_result_dict: a list of sentence result dictionary
        preprocess_result: a dictionary of keys: "tokens", "char2token_map", and "ner"
    '''
    document = NER_tagger(document)
    document_entities = filter_protein_entities(document.ents, document)

    # a list of list of tokens
    tokens = [str(tok) for tok in document]

    sentence_offsets = [sent[0].idx for sent in document.sents]
    # token index 2 entity map
    tokidx2ent_map = {token_idx: token.ent_type_ for token_idx, token in enumerate(document)}

    # create character offset to sentence-token index map
    char2doctoken_map = {tok.idx:doc_tok_id for doc_tok_id, tok in enumerate(document)}

    # ner for each sentence.
    # -sent.start to convert to sentence-level anntation
    ner = [[ent.start , ent.end-1, tokidx2ent_map[ent.start]] for ent in document.ents]

    # create mapping from starting character position to entity id
    ent_char2_id_map = {ent.start_char:f'T{idx+1}' for idx, ent in enumerate(document_entities)}
    
    with open(f'{output_dir}/{doc_id}.a1','w')as f:
        [f.write(f'{ent_char2_id_map[ent.start_char]}\tProtein {ent.start_char} {ent.end_char}\t{ent}\n') for ent in document_entities]
    doc_result_dict = []
    for sentence in document.sents:
        doc_result_dict.append(generate_sentence_instance(document, sentence, doc_id, ent_char2_id_map, corpus_proteinOrigIdBySpan))
    
    # gather all the preprocess result
    preprocess_result = {
        'ner':ner,
        'tokens':tokens,
        'char2doctoken_map':char2doctoken_map,
        'sentence_offsets':sentence_offsets
    }

    return doc_result_dict, preprocess_result


def preprocess_input(document, doc_id, output_dir):
    '''

    '''
    corpus_docId2OrigId = {doc_id: doc_id}
    corpus_proteinOrigIdBySpan = defaultdict(dict)
    all_result_dict = defaultdict(list)

    doc_result_dict, preprocess_result = process_document(document, doc_id, output_dir, corpus_proteinOrigIdBySpan)
    for sent_result_dict in doc_result_dict:
        # make sure this contains something
        if sent_result_dict is not None:
            for key, value in sent_result_dict.items():
                all_result_dict[key].append(value)
    all_result_dict['sample_ids'] = np.arange(len(all_result_dict['tokenized_ids'])).tolist()

    # create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(f'{output_dir}/{doc_id}.pkl','wb') as f:
        pickle.dump(all_result_dict, f, protocol = 4)

    with open(f'{output_dir}/{doc_id}_protIdBySpan.pkl','wb') as f:
        pickle.dump(corpus_proteinOrigIdBySpan, f, protocol = 4)

    with open(f'{output_dir}/{doc_id}_origIdById.pkl' ,'wb') as f:
        pickle.dump(corpus_docId2OrigId, f, protocol = 4)
    
    with open(f'{output_dir}/{doc_id}_preprocess_result.pkl','wb') as f:
        pickle.dump(preprocess_result, f, protocol = 4)


