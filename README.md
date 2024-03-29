# CASN
Code for BIBM 2021 paper "Co-Attentive Span Network with Multi-task learning for Biomedical Named Entity Recognition"

### Requirements
* Python 3.6
* PyTorch 1.9.0
*  transformers 2.1.1  
*  nni 2.4
  
### Prepare Models
all Datasets, we use [BioBERT-base](https://github.com/dmis-lab/biobert)

### Training Instructions
  * NCBI dataset Experiment  Search Space is found in `data/NCBI/NCBI_search_space.json` 
  * config the experiment in the config_remote.yml file.
  *  label Embedding in `data/NCBI/label_NCBI_768.npy` 
  
#### Examples
(1) load NCBI dataset:
```
python ./DataLoader/data_loader.py

```

(2) train the NCBI on train dataset, evaluate on dev dataset:
```
nnictl create --config ./config_remote.yml --port 9989
```
  

