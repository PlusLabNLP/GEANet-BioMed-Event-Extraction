import pickle
import sys, os
import types
import copy
import re
import pdb
import pickle
import tqdm
from collections import defaultdict, OrderedDict
import argparse
import pprint
import shutil

SIMPLE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization', 'Phosphorylation']
REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
BIND = ['Binding']



def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(list(range(r))):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def unmerging(d):
    '''
    d: a sentence. corpus_id, corpus_token, corpus_pos_tag, corpus_trigger_label, corpus_interaction_idx, corpus_interaction_label, corpus_span
    '''
    trigger_labels = d[3]
    int_idxs = d[4]
    int_labels = d[5]
    spans = d[6]
    tokens = d[1]
    
    # assert len(spans) == len(tokens)
    assert len(trigger_labels) == len(tokens)
    assert len(int_idxs) == len(int_labels), pdb.set_trace()
    
    arg_combs = []
    for idx in range(len(tokens)):
        # For predicted case, if the predicted trigger is None, then skip
        if trigger_labels[idx] in ['None', 'Protein']:
            continue
        # find outgoing edges for current tokens
        # out_ints: each element is like ((l_idx, r_idx), int_label)
        out_ints = [i for i in zip(int_idxs, int_labels) if i[0][0] == idx]
        if len(out_ints) == 0:
            continue
        # else, meaning current tokens have outgoing edges
        trigger_label = trigger_labels[idx]
        assert trigger_label != 'None'
        
        arg_combs.extend(get_valid_combination(trigger_label, out_ints))
        # pdb.set_trace()
    return arg_combs
def get_valid_combination(trigger_label, out_ints, ignore=True):
    '''
    get ALL valid combs, will apply heuristics to get final combs later
    '''

    # if ignore:
    #     # only keep 'Theme' and 'Cause'
    #     for i in range(len(out_ints)):
    #         if out_ints[i][1] not in ['Theme', 'Cause']:
    #             out_ints[i][1] = 'None'

    # TODO: now only consider Theme and Cause
    # we might need to consider other args too in the future

    intsByType = defaultdict(list)

    # intsByType: { 'int_label': (int_idxs, int_label)}
    for i in out_ints:
        intsByType[i[1]].append(i)

    arg_combs = []
    if trigger_label in SIMPLE:
        if ignore:
            # only one argument: (Theme, )
            # arg_combs.extend(intsByType['Theme'])
            for i in combinations(intsByType['Theme'], 1):
                arg_combs.append(i)
    elif trigger_label in BIND:
        if ignore:
            # arbitrary number of Theme args
            # Note that TEES maps all Theme2, Theme3 to Theme
            max_len = len(intsByType['Theme'])
            for length in range(1, max_len+1):
                for i in combinations(intsByType['Theme'], length):
                    arg_combs.append(i)
    elif trigger_label in REG:
        if ignore:
            # first generate all Theme-only args
            for i in combinations(intsByType['Theme'], 1):
                # first find (Theme, ) combinations
                arg_combs.append(i)
            if 'Cause' in intsByType:
                # NOTE: Cause-only edges are considered as non-event, hence not valid
                # for i in combinations(intsByType['Cause'], 1):
                #     # first find (Cause, ) combinations
                #     # we do this is because there are some inter-sent cases will miss the Theme edge, but leave the Casue edge
                #     # although this is not a valid event but the Cause edge should be helpful training signal and we keep them
                #     arg_combs.append(i)
                # then also find (Theme, Cause) combinations
                for i in combinations(intsByType['Theme'], 1):
                    for j in combinations(intsByType['Cause'], 1):
                        arg_combs.append((i[0], j[0]))
    else:
        
        raise NotImplementedError
    return arg_combs
def heu_for_unmerge(arg_combs, d):
    '''
    Apply heuristics for different event types here, e.g. when both (Theme, Cause) and (Theme, ) combinations appear for current token
    Only keep the longest chain, i.e. (Theme, Cause)
    params:
        d: sent
        arg_combs: ALL valid combs of this sent
    '''
    final_combs = []
    trigger_types = d[3]
    
    for idx in range(len(trigger_types)):
        trigger_label = trigger_types[idx]
        cur_combs = [i for i in arg_combs if i[0][0][0] == idx]
        if trigger_label in ['None', 'Protein', 'Entity']:
            continue
        # inter-sent interactions, meaning a trigger no outgoing edges, skip
        if len(cur_combs) == 0:
            continue

        if trigger_label in SIMPLE:
            
            assert max([len(i) for i in cur_combs]) == 1
            # do nothing for SIMPLE types, because they all have only one Theme
            # final_combs.extend(cur_combs)
        elif trigger_label in BIND:
            # pass
            max_chain_len = max([len(i) for i in cur_combs])
            # only keep longest chain like (Theme, Theme) if any
            # if max_chain_len < 4:
            cur_combs = [i for i in cur_combs if len(i) == max_chain_len]
            # else:
                # cur_combs = [i for i in cur_combs if len(i) in [1, 2, 3]]
        elif trigger_label in REG:
            max_chain_len = max([len(i) for i in cur_combs])
            # only keep longest chain like (Theme, Cause) if any
            cur_combs = [i for i in cur_combs if len(i) == max_chain_len]
        final_combs.extend(cur_combs)
    return final_combs

def is_terminal(event):
    
    if 'Theme' not in event:
        return False
    targets = []
    for k, v in list(event.items()):
        if k.startswith('Theme') or k == 'Cause':
            targets.append(v)
    # print(targets)
    return all([i.startswith('T') for i in targets])


def map_theme_for_bind(cur_comb):
    '''
    Since TEES maps all Theme2, Theme3 args to Theme
    We now convert it back. o.w. there will be unexpected erorwhen constructing the event vocabulary
    since all Theme2, Theme3 will share the same Theme key
    '''
    edge_labels = [i[1] for i in cur_comb]
    assert len(set(edge_labels)) == 1
    if len(cur_comb) == 1:
        return cur_comb
    assert len(cur_comb) > 1 # it must have more than one Theme edge
    out_comb = []
    theme_id = 1
    for edge in cur_comb:
        out_edge = []
        out_edge.append(edge[0])
        if theme_id == 1:
            out_edge.append(edge[1])
        else:
            out_edge.append('{}{}'.format(edge[1], theme_id))
        theme_id +=1
        out_comb.append(tuple(out_edge))
    return out_comb


def writeA2(orig_docid, args, triggerIdBySpan, triggerTypeBySpan, all_events):
    triggerIdBySpan = OrderedDict(sorted(list(triggerIdBySpan.items()), key=lambda x:int(x[0].split('-')[0])))
    print(all_events)
    with open(f'{args.input_dir}/{orig_docid}.txt','r') as f:
        origial_texts = f.read()

    # gather all events
    all_events = sorted(all_events, key=lambda x: int(x['ST_id'][1:]))
    
    # if nothing to write, return
    if len(all_events) == 0 and len(triggerIdBySpan) == 0: return
    
    with open('{}/{}.a2'.format(args.out_dir, orig_docid), 'w') as f:
        # first write triggers
        for span, trigger_id in list(triggerIdBySpan.items()):
            span_l = span.split('-')[0]
            span_r = span.split('-')[1]
            T_line = '{}\t{} {} {}\t{}\n'.format(trigger_id, triggerTypeBySpan[span], span_l, span_r, origial_texts[int(span_l):int(span_r)])
            f.write(T_line)

        
        # then write events
        for event in sorted(all_events, key=lambda event: triggerIdBySpan[event['trigger_span']] ):
            
            arg_str = ' '.join([':'.join([k, v]) for (k, v) in sorted(list(event.items()), reverse=True, key=lambda x: x[0]) if k.startswith('Theme') or k == 'Cause'])
            E_line = '{}\t{}:{} '.format(event['ST_id'], event['trigger_type'], triggerIdBySpan[event['trigger_span']]) + \
                     arg_str + '\n'
            f.write(E_line)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_newevent(event):
    if 'Cause' in event:
        if (not event["Cause"].startswith('T')) and (not event['Cause'].startswith('E')):
            return False
    for k, v in list(event.items()):
        if k.startswith('Theme'):
            if (not event[k].startswith('T')) and (not event[k].startswith('E')):
                return False
    return True

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-apply_heu', type=str2bool, default=True, help='apply the heuristics of taking the longest chain for unmerging')
    p.add_argument('-pred_pkl', type=str, default='CORD_out_merged.pkl')
    p.add_argument('-protIdBySpan', type=str, default='preprocessed_data/CORD_19_PMCprotIdBySpan.pkl')
    p.add_argument('-origIdById', type=str, default='preprocessed_data/CORD_19_PMCorigIdById.pkl')
    p.add_argument('-out_dir', type=str, default='genia_cord_19_output')
    p.add_argument('-input_dir', type=str, default='genia_cord_19')
    args = p.parse_args()
    


    # with open('GE11_dev_flat_w-span_w-dep_NoEntSite.pkl', 'r') as f:
    # with open('../out_pkl_1120_fulldata/GE11_dev_EXP013_Tf10.6629_If10.7924_pred-w-goldTrue_pred-w-predFalse_lr0.001_drp0.5_ep9_hd70_seed42_tw1.0_iw0.5_trEmbFalse_twNoneTrue.pkl', 'r') as f:
    # with open('../out_pkl_1120_fulldata/GE11_dev_EXP009_Tf10.6408_If10.8017_pred-w-goldTrue_pred-w-predFalse_lr0.001_drp0.1_ep9_hd70_seed42_tw1.0_iw0.5_trEmbFalse_twNoneFalse.pkl', 'r') as f:
    # with open('../out_pkl_1120_fulldata/GE11_dev_EXP083_Tf10.6360_If10.8137_pred-w-goldTrue_pred-w-predFalse_lr0.001_drp0.1_ep27_hd100_seed42_tw1.0_iw1.0_trEmbFalse_twNoneFalse.pkl', 'r') as f:
    # with open('../out_pkl_1120_fulldata/GE11_dev_EXP062_Tf10.6627_If10.7959_pred-w-goldTrue_pred-w-predFalse_lr0.001_drp0.5_ep11_hd100_seed42_tw1.0_iw0.5_trEmbFalse_twNoneTrue.pkl', 'r') as f:
    # with open('../out_pkl_1122/GE11_dev_EXP128_Tf10.6781_If10.8063_pred-w-goldTrue_pred-w-predTrue_lr0.001_drp0.2_ep10_hd100_seed42_tw1.0_iw1.0_trEmbFalse_twNoneTrue_iwNoneFalse_UPFalse.pkl', 'r') as f:
    ######### this one gives 47.67
    # with open('../out_pkl_1122/GE11_dev_EXP160_Tf10.6563_If10.7794_pred-w-goldFalse_pred-w-predTrue_lr0.001_drp0.3_ep26_hd40_seed42_tw1.0_iw2.0_trEmbFalse_twNoneTrue_iwNoneFalse_UPFalse.pkl', 'r') as f:
    # with open('../out_pkl_1122/GE11_dev_EXP224_Tf10.6627_If10.7516_pred-w-goldFalse_pred-w-predTrue_lr0.001_drp0.4_ep23_hd40_seed42_tw1.0_iw1.0_trEmbFalse_twNoneTrue_iwNoneFalse_UPFalse.pkl', 'r') as f:
    # with open('../out_pkl_1122/GE11_dev_EXPtest_Tf10.6723_If10.7979_pred-w-goldTrue_pred-w-predTrue_lr0.001_drp0.2_ep7_hd100_seed42_tw1.0_iw1.0_trEmbFalse_twNoneTrue_iwNoneFalse_UPFalse.pkl', 'r') as f:
    # with open('../out_pkl_1125/GE11_test_EXP120_Tf10.6817_If10.7494_pred-w-goldFalse_pred-w-predTrue_lr0.05_drp0.5_ep18_hd100_seed42_tw1.0_iw1.0_trEmbFalse_twNoneTrue_iwNoneFalse_UPFalse.pkl', 'r') as f:
    # with open('../out_pkl_1120_fulldata/GE11_dev_fortest.pkl', 'r') as f:
    # with open('../EMNLP-2019/code/GE11_dev_out_goldT.pkl', 'r') as f:
    with open(args.pred_pkl, 'rb') as f:
        data = pickle.load(f)
    with open(args.protIdBySpan, 'rb') as f:
        # NOTE: the protIdBySpan is already processed by TEES
        # this is not the gold annotation from a1 files
        protIdBySpan = pickle.load(f)
    with open(args.origIdById, 'rb') as f:
        origIdById = pickle.load(f)

    # pdb.set_trace()

    # clear out everything
    
    # shutil.rmtree(args.out_dir, ignore_errors=True)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print('Unmerging ...')
    for i in tqdm.tqdm(list(range(len(data)))):
        arg_combs = unmerging(data[i])

        # pdb.set_trace()
        if len(arg_combs) != 0:
            # TODO: this is an error !!!!!!!!!
            # this will lose events from Gene_exp when there are both REG and Gene_exp
            # max_chain_len = max([len(a) for a in arg_combs])
            # when there are both (Theme, Cause) and (Theme), choose the longer one, i.e. (Theme, Cause)
            # final_combs save the final seletected combinations by our rules
            # these are actuallly the unmerged events
            # final_combs = [a for a in arg_combs if len(a) == max_chain_len]
            if args.apply_heu:
                final_combs = heu_for_unmerge(arg_combs, data[i])
            else:
                final_combs = arg_combs
            # pdb.set_trace()
            data[i] += tuple([final_combs], )
        else:
            data[i] += tuple([[]], )

        # print(data[i][0])
        # if data[i][0] == 'GE09.d167.s1':
        #     pdb.set_trace()
        # if data[i][0] == 'GE09.d167.s2':
        #     pdb.set_trace()
    # pdb.set_trace()
    

    print('Finding events ...')
    for doc_id in list(protIdBySpan.keys()): # doc_id: GE09.d1
        # try:
        doc_sents = [i for i in data if i[0] == doc_id]
        
        # sort the doc with sentence order
        doc_sents = sorted(doc_sents, key=lambda x: int(x[0].split('.')[-1][1:]))

        doc_prots = protIdBySpan[doc_id]    # doc_prots: e.g. '1-3':'10086.T1'
        
        # print(origIdById)
        orig_docid = origIdById[doc_id]
        # orig_docid = doc_prots.values()[0].split('.')[0]
        if len(list(doc_prots.values())) == 0:
            # doc does not have Proteins
            f = open('{}/{}.a2'.format(args.out_dir, orig_docid), 'w')
            f.close()
            continue
        # orig_docid = doc_prots.values()[0].split('.')[0]
        # find the proteins annotated in .a1 file
        # then decide the start id of trigger
        
        trigger_id = 1 + max([int(i.split('.')[-1][1:]) for i in list(doc_prots.values())])

        # event_id = 1
        triggerIdBySpan = {}
        triggerTypeBySpan = {}
        tokenBySpan = {}
        eventIdsBySpan = defaultdict(list)
        # eventSTById = defaultdict(dict)

        # eventsBySpan = defaultdict(list)
        events = []
        event_id = 1
        # if orig_docid =='PMC1074505.d2':
        #     print(doc_prots)
        #     print(trigger_id)
        
        for d in doc_sents:
            trigger_types = d[3]
            spans = d[6]
            
            tokens = d[1]
            int_idxs = d[4]
            int_labels = d[5]
            arg_combs = d[-1]
            
            
            # from collections import Counter
            # print(Counter(trigger_types))
            assert len(trigger_types) == len(spans)
            assert len(spans) == len(tokens)

            for idx in range(len(trigger_types)):
                if trigger_types[idx] in ['Protein', 'Entity', 'None']:
                    # skip triggers that are not events
                    # these include None, Protein, Entity
                    continue
                if len([i for i in arg_combs if i[0][0][0] == idx]) == 0:
                    # no valid event for this trigger
                    continue
                
                assert spans[idx] not in triggerIdBySpan
                triggerIdBySpan[spans[idx]] = 'T{}'.format(trigger_id)
                triggerTypeBySpan[spans[idx]] = trigger_types[idx]
                tokenBySpan[spans[idx]] = tokens[idx]
                trigger_id += 1
            
            # pdb.set_trace()
            ############ step1: find all terminal events (leaf)
            for idx in range(len(trigger_types)):
                cur_combs = [i for i in arg_combs if i[0][0][0] == idx]
                # if spans[idx] == '':
                #     pdb.set_trace()
                
                for cur_comb in cur_combs:
                    event = {}
                    if trigger_types[idx] == 'Binding':
                        # map theme back to theme2 ... for Binding event
                        cur_comb = map_theme_for_bind(cur_comb)
                    event['trigger_span'] = spans[idx]
                    event['trigger_type'] = trigger_types[idx]
                    
                    for edge in cur_comb:
                        assert edge[0][0] == idx
                        target_idx = edge[0][1]
                        target_span = spans[target_idx]
                        arg_role = edge[1]
                        
                        if target_span in triggerIdBySpan and target_span in doc_prots:
                            # it is a nesting parent, its target is presented by span for now
                            # for predicted case, a protein location can be predicted as event
                            # we select protein instead
                            
                            event[arg_role] = doc_prots[target_span].split('.')[-1]
                        elif target_span in doc_prots:
                            # it points to Protein, replace with the protein ID
                            event[arg_role] = doc_prots[target_span].split('.')[-1]
                        elif target_span in triggerIdBySpan:
                            event[arg_role] = target_span
                        else:
                            # wrong prediction, in gold the target_span should be a event trigger but the predicted target_span position
                            # is None(not picked up by model, then this target_span will not be present in triggerIdBySpan, also not in doc_prots
                            # or the predicted target_span corresponds to a non-valid event(e.g. gene_exp with inter-sent protein theme)
                            continue
                            # pass

                    # if d[0] == 'GE09.d235.s11':
                    #     pdb.set_trace()
                    # print(cur_comb, is_terminal(event))
                    if is_terminal(event):
                        event['ST_id'] = 'E{}'.format(event_id)  # Assign event ids for terminal events
                        eventIdsBySpan[spans[idx]].append(event['ST_id'])
                        event_id += 1
                    else:
                        event['ST_id'] = 'X'  # Nesting events, Id To be determined later
                    # Think about this!!!!!!!!!!!
                    # if 'Theme' not in event and 'Cause' in event:
                    #     pdb.set_trace()
                    # NOTE: need to think about this !!!!!!!!!!!!!!!!!!!
                    # is it valid to have this artificial screening?
                    # predicted event might not have a theme edge, see above
                    # pdb.set_trace()
                    if not 'Theme' in event:
                        continue
                    events.append(event)


        ######### step2: find all nesting events via maintaining a stack
        event_cand_stack = [i for i in events if i['ST_id'] == 'X']
        new_events = []
        # print(event_cand_stack)
        # if len(event_cand_stack) > 0:
        #     pdb.set_trace()
        # print(event_id)
        # if orig_docid == 'PMID-8895544':
        #     pdb.set_trace()
        while event_cand_stack:
            remove = [False] * len(event_cand_stack)
            # pre_len = len(remove)  #record state of remove mask
            for idx in range(len(event_cand_stack)):
                cur_event = event_cand_stack[idx]
                # NOTE: Need to think about it!!!!!!!!!!!
                # mis-classified event trigger type, the predicted events might say a Gene_expression will also nests other events
                # so this assertion is commented for now
                # TODO: this assertion does not hold for predicted case - WHY??????
                # assert cur_event['trigger_type'] in REG, pdb.set_trace()
                try:
                    theme_target_span = cur_event['Theme']
                except:
                    pdb.set_trace()
                cause_target_span = cur_event.get('Cause', None)

                # pdb.set_trace()
                if cause_target_span:
                    if cause_target_span.startswith('T'):
                        # a protein
                        cause_target_ids = [cause_target_span]
                    else:
                        cause_target_ids = eventIdsBySpan.get(cause_target_span, None)
                else:
                    cause_target_ids = None
                if theme_target_span.startswith('T'):
                    theme_target_ids = [theme_target_span]
                else:
                    theme_target_ids = eventIdsBySpan.get(theme_target_span, None)
                if cause_target_span:
                    if theme_target_ids is not None and cause_target_ids is not None:
                        # both theme and cause point to (known) child trigger
                        new_combs = [(x, y) for x in theme_target_ids for y in cause_target_ids]
                        # TODO: this is a simple heuristic to avoid overly dense output
                        # need to reengineer later
                        # if len(new_combs) >= 8:
                        #     continue
                        for i in range(len(new_combs)):
                            new_event = {}
                            new_event['trigger_type'] = cur_event['trigger_type']
                            new_event['trigger_span'] = cur_event['trigger_span']
                            assert cause_target_span
                            new_event['Cause'] = new_combs[i][1]
                            new_event['Theme'] = new_combs[i][0]
                            new_event['ST_id'] = 'E{}'.format(event_id)
                            eventIdsBySpan[cur_event['trigger_span']].append(new_event['ST_id'])
                            event_id += 1
                            new_events.append(new_event)
                            # the parent events have been found and added
                            remove[idx] = True
                            if not check_newevent(new_event):
                                pdb.set_trace()
                else:
                    # no cause arg
                    if theme_target_ids is not None:
                        # only theme point to a (known) child event trigger
                        # TODO: this is a simple heuristic to avoid overly dense output
                        # need to reengineer later
                        # if len(theme_target_ids) >= 8:
                        #     continue
                        for i in range(len(theme_target_ids)):
                            new_event = {}
                            new_event['trigger_type'] = cur_event['trigger_type']
                            new_event['trigger_span'] = cur_event['trigger_span']
                            # if cause_target_span:
                            #     new_event['Cause'] = cur_event['Cause']
                            new_event['Theme'] = theme_target_ids[i]
                            new_event['ST_id'] = 'E{}'.format(event_id)
                            eventIdsBySpan[cur_event['trigger_span']].append(new_event['ST_id'])
                            event_id += 1
                            new_events.append(new_event)
                            # the parent events have been found and added
                            remove[idx] = True
                            if not check_newevent(new_event):
                                pdb.set_trace()
                # elif cause_target_ids is not None and theme_target_ids is None:
                #     # only Cause point to a (known) child event trigger
                #     for i in range(len(cause_target_ids)):
                #         new_event = {}
                #         new_event['trigger_type'] = cur_event['trigger_type']
                #         new_event['trigger_span'] = cur_event['trigger_span']
                #         if theme_target_span:
                #             new_event['Theme'] = cur_event['Theme']
                #         new_event['Cause'] = cause_target_ids[i]
                #         new_event['ST_id'] = 'E{}'.format(event_id)
                #         eventIdsBySpan[cur_event['trigger_span']].append(new_event['ST_id'])
                #         event_id += 1
                #         new_events.append(new_event)
                #         # the parent events have been found and added
                #         remove[idx] = True
                #         if not check_newevent(new_event):
                #             pdb.set_trace()
                # else:
                #     # target spans are unknown, meaning the child is not known yet
                #     continue
                # if orig_docid == 'PMID-8895544':
                #     pdb.set_trace()
            event_cand_stack = [event_cand_stack[i] for i in range(len(event_cand_stack)) if remove[i] == False]
            if set(remove) == set([False]): #and len(remove) == prev_len:
                # found the root(s), no more update
                break
        # print(len(event_cand_stack), len([i for i in event_cand_stack if i['ST_id'] == 'X']))
        # if len(event_cand_stack) > 0 :
        #     pdb.set_trace()
        
        all_events = [event for event in events+new_events if event['ST_id'] != 'X']
        
        for event in all_events:
            if event['trigger_type'] not in REG:
                assert 'Cause' not in event
                for k,v in list(event.items()):
                    if k.startswith('Theme'):
                        assert event[k].startswith('T') or event[k].startswith('E')
            if (not event['Theme'].startswith('T')) and (not event['Theme'].startswith('E')):
                assert event['trigger_type'] in REG, pdb.set_trace()
            for k, v in list(event.items()):
                if k.startswith('Theme') or k == 'Cause':
                    assert event[k] not in eventIdsBySpan
                # pdb.set_trace()

        
        
        writeA2(orig_docid, args, triggerIdBySpan, triggerTypeBySpan, all_events)
        # except Exception as e:
        #     print(e)
