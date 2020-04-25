#!/bin/bash
task="GE"
s=(4)
mlp_hid_size=64
seed=42
model="scibert"
prefix="pipeline"
pw=2.0
gpu=0
# pipe_epoch=5
do_train=false
do_eval=false
do_test=true

nt_cls=false
mnc=false
link_pred=false
edge_cls=false


test_kg_datas=test_kg_datas #_split_transe_mst_th23000

use_knowledge=false
use_temporal_edge=false
kg_pretrained_weights=pretrained_kge/transe_margin_d300_adam-0.5_t500.ckpt #pretrained_kge/UMLS_CUI_STY_temporal_transe_margin_d300_adam-0.5_t350.ckpt #pretrained_kge/UMLS_CUI_STY_split_transe_margin_d300_adam-0.5_t300.ckpt

gnn_type=ECGAT
pepochs=100
ent_linear_size=(1000)
rel_linear_size=(300)
e=(100)
dp=(0.1)
kg_embedding_dim=(300)
lr=3e-5
output_dir=weights/pipeline_scibert_batch_4_lr_3e-5_epochs100_pepochs100_seed_42_dp0.1_know-false_kg_emb300_ent1000_rel300

python run_multitask_bert.py -gnn_type=${gnn_type} -use_temporal_edge=${use_temporal_edge} -use_knowledge=${use_knowledge} -ent_linear_size=${ent_linear_size} -rel_linear_size=${rel_linear_size} -kg_pretrained_weights=${kg_pretrained_weights} -kg_embedding_dim=${kg_embedding_dim} -test_kg_datas=${test_kg_datas} -dropout=${dp} -edge_cls=${edge_cls} -link_pred=${link_pred} -nt_cls=${nt_cls} -mnc=${mnc} -model=${model} -random_seed=${seed} -train_batch_size ${s} -lr ${lr} -pipe_epoch=${e} -epochs=${e}   -regen_vocfile=True -data_dir=preprocessed_data  -output_dir=${output_dir} -gpu=${gpu} -do_eval=${do_eval} -do_train=${do_train} -do_test=${do_test}
