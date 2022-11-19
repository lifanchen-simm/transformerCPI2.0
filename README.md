# TransfomerCPI2.0

  We only disclose the inference model.

## Setup and dependencies 
`environment.yaml` is the conda environment of this project.

## Inference
`predict.py` makes the inference, the input are protein sequence and compound SMILES. `featurizer.py` tokenizes and encodes the protein sequence and compounds. `mutation_analysis.py` conducts drug mutation analysis to predict binding sites. `substitution_analysis.py` conducts substitution analysis.

## Trained models
Trained models will be available at https://drive.google.com/drive/folders/1X7i1eO-EykCQcvqMeWeB7QXT3E9eLG08?usp=sharing or https://jianguoyun.simm.ac.cn/p/DYIrdecQUxifFA. Trained model will be released after the article is accepted or another time in the future.

## Requirements
python = 3.8.8 

pytorch = 1.9 

tape-proteins = 0.5 

rdkit = 2021.03.5 

numpy = 1.19.5 

scikit-learn = 0.24.1 

