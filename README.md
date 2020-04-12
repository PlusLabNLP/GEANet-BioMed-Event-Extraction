# Biomedical Event Extraction on CORD-19

<a href="https://pluslabnlp.github.io/"><img src="https://pluslabnlp.github.io/images/Logos/logo_transparent_background.png" height="120" ></a>
<a href="https://www.isi.edu/"><img src="https://pluslabnlp.github.io/images/usc-logo.png"  height="120"></a>


## Environment
* CentOS Linux 7
* CUDA 10.1 
* Python==3.6.10
* PyTorch==1.4.0 
* Numpy==1.16
* Pandas==0.24.2
* Sklearn==0.23.0
* Scipy==1.4.1


## Instructions

1. Download the [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) parallel to the current directory.
2. Run through `ReparseCORD.ipynb` to generate reparsed .txt files to `genia_cord_19` from the json files in the `custom_license/custom_license/pmc_json/` directory.

