# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/4/27 13:56
@author: LiFan Chen
@Filename: mutation_analysis.py
@Software: PyCharm
"""

import torch
import random
import os
from model import *
import numpy as np
from featurizer import featurizer

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    # device = torch.device('cpu')
    """ create model ,trainer and tester """
    n_layers = 3
    dropout = 0.1
    batch = 64
    lr = 1e-5
    weight_decay = 1e-3
    decay_interval = 10
    lr_decay = 1.0
    iteration = 100

    pretrain = torch.load('Bert.pkl')
    pretrain.to(device)
    for param in pretrain.parameters():
        param.requires_grad = False
    pretrain.eval()
    encoder = Encoder(pretrain,n_layers,device)
    decoder = Decoder(n_layers, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load("lr=1e-5,weight_decay=1e-3,dropout=0.1,batch=64.pt",map_location=lambda storage, loc: storage))
    model.to(device)
    tester = Tester(model,device)


    """Prepare input data. Including SMILES, Sequence and Interaction"""
    """Start training."""
    print('Testing...')
    sequence = "MSCAGRAGPARLAALALLTCSLWPARADNASQEYYTALINVTVQEPGRGAPLTFRIDRGRYGLDSPKAEVRGQVLAPLPLHGVADHLGCDPQTRFFVPPNIKQWIALLQRGNCTFKEKISRAAFHNAVAVVIYNNKSKEEPVTMTHPGTGDIIAVMITELRGKDILSYLEKNISVQMTIAVGTRMPPKNFSRGSLVFVSISFIVLMIISSAWLIFYFIQKIRYTNARDRNQRRLGDAAKKAISKLTTRTVKKGDKETDPDFDHCAVCIESYKQNDVVRILPCKHVFHKSCVDPWLSEHCTCPMCKLNILKALGIVPNLPCTDNVAFDMERLTRTQAVNRRSALGDLAGDNSLGLEPLRTSGISPLPQDGELTPRTGEINIAVTKEWFIIASFGLLSALTLCYMIIRATASLNANEVEWF"
    smiles = "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O"
    compounds, adjacencies, proteins = featurizer(smiles, sequence)
    test_set = list(zip(compounds, adjacencies, proteins))
    original_score = tester.test(test_set)
    mutation = 'ARNDCQEGHILKMFPSTWYV'
    n = len(sequence)
    delta_score = np.zeros((n,20))
    for i in range(n):
        k = 0
        for m in mutation:
            sequence_2 = list(sequence)
            sequence_2[i] = m
            sequence_2 = ''.join(sequence_2)
            compounds,adjacencies,proteins = featurizer(smiles,sequence_2)
            test_set = list(zip(compounds,adjacencies,proteins))
            score = tester.test(test_set)
            delta_score[i,k] = np.abs(original_score - score)
            print(delta_score[i,k])
            k += 1
    np.save('mutation_RNF130.npy',delta_score)
