# Biomedical Event Extraction on CORD-19

<a href="https://pluslabnlp.github.io/"><img src="https://pluslabnlp.github.io/images/Logos/logo_transparent_background.png" height="120" ></a>
<a href="https://www.isi.edu/"><img src="https://pluslabnlp.github.io/images/usc-logo.png"  height="120"></a>



## Introduction
With the increasing concern about the COVID-19 pandemic, researchers have been putting much effort in providing useful insights into the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/). This repo demonstrates how we extract biomedical events with [SciBERT](https://github.com/allenai/scibert), a BERT trained on scientific corpus, which was fine-tuned on the [GENIA BioNLP shared task 2011](http://2011.bionlp-st.org/home/genia-event-extraction-genia). We took reference from the described in pipeline *Generalizing Biomedical Event Extraction (Bjorne et al.)*, where the pipeline can be broken into 3 stages: trigger detection, edge detection and unmerging. The extracted events can be found [here](https://drive.google.com/file/d/1FXN2QRBoFzQmLwQztUhULm8WVKxyRwu3/view?usp=sharing).

<p align="center"><img src="https://github.com/jbjorne/TEES/wiki/TEES-process.png"   style="margin:auto"></p>

In addition, we adopted the framework of *Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction (Han et al.)*, where trigger and edge detections are trained in a multitask setting.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1e60ed5b66f1cb8df30e77820f53a036c2d35d3c/3-Figure2-1.png"   style="margin:auto"></p>

## Instructions


1. Download the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) parallel to the current directory.
2. Get UMLS permission and place the `MRSTY.RRF` file in the `umls_data` folder.
3. Run through `Preprocess CORD.ipynb` to generate the processed .txt files to `genia_cord_19` from the json files in the `custom_license/custom_license/pmc_json/` directory.
4. Download the model's [weights](https://drive.google.com/file/d/1GswpExncD4t5WAVijvh5c8Vtd4KGpH9U/view?usp=sharing) and extract it.
5. Type `. run_multitask_bert.sh` in the terminal to run the whole event extraction pipeline and generate event annotations into `genia_cord_19_output`.


## Environment & Packages
* CentOS Linux 7
* CUDA 10.1 
* Python==3.6.10
* PyTorch==1.4.0 
* Numpy==1.16
* Pandas==0.24.2
* Sklearn==0.23.0
* Scipy==1.4.1
* Joblib==0.14.1
* Transformers==2.5.0
* Spacy==2.2.4
* Scispacy

## Project Structure

```
├── genia_cord_19
├── genia_cord_19_output
├── umls_data
├── preprocessed_data
├── eval
└── weights

```

## Performance
Currently, this repo contains only the fine-tuned SciBERT on the GENIA dataset, whose performance is on par with the previous SOTA result. The best performing model with UMLS (a large biomedical knowledge grapah) and GNN incorporation will soon be released.
| Model        | Dev Set F1           | Test Set F1  |
| ------------- |-------------:| -----:|
|   SciBERT Baseline    | 59.33      |   58.50  |
|   SciBERT w/ UMLS & GNN (coming soon)   | **60.38** | **60.06** |
| [Previous SOTA](https://www.aclweb.org/anthology/N19-1145.pdf) | N/A      |   58.65  |