# TransfomerCPI2.0

  We only disclose the inference models. TransformerCPI2.0 is based on TransformerCPI whose codes are all released. The details of TransformerCPI2.0 are described in our paper https://www.biorxiv.org/content/10.1101/2022.03.26.485909v4. Trained models are not available at present.

## Setup and dependencies 
`environment.yaml` is the conda environment of this project.

## Inference
`predict.py` makes the inference, the input are protein sequence and compound SMILES. `featurizer.py` tokenizes and encodes the protein sequence and compounds. `mutation_analysis.py` conducts drug mutation analysis to predict binding sites. `substitution_analysis.py` conducts substitution analysis.

## Trained models
Trained models will be available at https://drive.google.com/drive/folders/1X7i1eO-EykCQcvqMeWeB7QXT3E9eLG08?usp=sharing. Trained models may be released after the article is accepted or an appropriate time in the future (usage need permission).

## Requirements
python = 3.8.8 

pytorch = 1.9 

tape-proteins = 0.5 

rdkit = 2021.03.5 

numpy = 1.19.5 

scikit-learn = 0.24.1 

