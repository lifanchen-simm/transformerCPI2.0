# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/4/29 10:57
@author: LiFan Chen
@Filename: substitution_analysis.py
@Software: PyCharm
"""
import torch
import random
import os
import numpy as np
from featurizer import featurizer
import pandas as pd

if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    model = torch.load('TransformerCPI2.0.pt')  # Load Trained model
    model.to(device)
    
    """Prepare input data. Including SMILES, Sequence and Interaction"""
    df = pd.read_csv('TFMvsMePairs.csv',sep='\t')
    with open('result.txt','w') as f:
        f.write('{}\n'.format('Target_ID,Uniprot_ID,-CF3_ID,-CH3_ID,delta_pAct,delta_score'))
        for i in range(len(df)):
            target_id = df.iloc[i]['Target_ID']
            uniprot_id = df.iloc[i]['Target_uniprot']
            cf3_id = df.iloc[i][' -CF3_ID']
            ch3_id = df.iloc[i][' -CH3_ID']
            delta_pAct = df.iloc[i]['delta_pAct']
            sequence = df.iloc[i]['uniprot_fasta']
            smiles_cf3 = df.iloc[i][' -CF3_smiles']
            try:
                compounds,adjacencies,proteins = featurizer(smiles_cf3,sequence)
            except:
                continue
            test_set = list(zip(compounds,adjacencies,proteins))
            score_cf3 = float(tester.test(test_set))
            smiles_ch3 = df.iloc[i][' -CH3_smiles']
            compounds, adjacencies, proteins = featurizer(smiles_ch3, sequence)
            test_set = list(zip(compounds, adjacencies, proteins))
            score_ch3 = float(tester.test(test_set))
            delta_score = score_cf3 - score_ch3
            f.write('{}\n'.format(','.join([str(target_id),str(uniprot_id),str(cf3_id),str(ch3_id),str(delta_pAct),str(delta_score)])))
            f.flush()