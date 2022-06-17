# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/10/13 13:54
@author: LiFan Chen
@Filename: predict.py
@Software: PyCharm
"""

import torch
import random
import os
import numpy as np

class Tester(object):
    def __init__(self, model,device):
        self.model = model
        self.device = device
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins = [], [], []
                atom, adj, protein= data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                data = pack(atoms,adjs,proteins, self.device)
                predicted_scores = self.model(data)
        return predicted_scores

def pack(atoms, adjs, proteins, device):
    atoms = torch.FloatTensor(atoms)
    adjs = torch.FloatTensor(adjs)
    proteins = torch.FloatTensor(proteins)
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0]+1)
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atoms_len += 1
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, 1:a_len+1, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        adjs_new[i,0,:] = 1
        adjs_new[i,:,0] = 1
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, 1:a_len+1, 1:a_len+1] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len),dtype=torch.int64, device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)


if __name__ == "__main__":
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    model = torch.load('TransformerCPI2.0.pt')  # Load trained model
    model.to(device)
    sequence = "MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA" # Example protein sequence
    smiles = "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O" # Example compound
    compounds, adjacencies, proteins = featurizer(smiles, sequence)
    tester = Tester(model, device)
    test_set = list(zip(compounds, adjacencies, proteins))
    score = float(tester.test(test_set))
    print(score)
