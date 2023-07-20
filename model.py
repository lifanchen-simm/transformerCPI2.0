# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/4/27 13:41
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, pretrain,n_layers, device):
        super().__init__()
        self.pretrain = pretrain
        self.hid_dim = 768
        self.n_layers = n_layers
        self.device = device
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim*4,dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, protein, mask):
        # protein = [batch, seq len]
        # mask = [batch, seq len]  0 for true positions, 1 for mask positions
        input_mask = (mask > 0).float()
        with torch.no_grad():
            protein = self.pretrain(protein,input_mask)[0]
        protein = protein.permute(1,0,2).contiguous() # protein = [seq len, batch, 768]
        mask = (mask == 1)
        protein = self.encoder(protein,src_key_padding_mask=mask)
        # protein = [seq len, batch, 768]
        return protein, mask


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self,n_layers,dropout,device):
        super().__init__()

        self.device = device
        self.hid_dim = 768
        self.n_layers = n_layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim * 4,dropout=0.1)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.fc_1 = nn.Linear(768, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, hid_dim]
        # src = [protein len, batch, hid_dim] # encoder output
        trg = trg.permute(1,0,2).contiguous()
        # trg = [compound len, batch, hid_dim]
        trg_mask = (trg_mask == 1)
        trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
        # trg = [compound len,batch size, hid dim]
        trg = trg.permute(1,0,2).contiguous()
        # trg = [batch, compound len, hid dim]
        x = trg[:,0,:]
        label = F.relu(self.fc_1(x))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.fc_1 = nn.Linear(atom_dim, atom_dim)
        self.fc_2 = nn.Linear(atom_dim, 768)

    def gcn(self, input, adj):
        # input =[batch,num_node, atom_dim]
        # adj = [batch,num_node, num_node]
        support = self.fc_1(input)
        # support =[batch,num_node,atom_dim]
        output = torch.bmm(adj, support)
        # output = [batch,num_node,atom_dim]
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len,device):
        N = len(atom_num)  # batch size
        compound_mask = torch.ones((N, compound_max_len),device=device)
        protein_mask = torch.ones((N, protein_max_len),device=device)
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 0
            protein_mask[i, :protein_num[i]] = 0
        return compound_mask, protein_mask


    def forward(self, compound, adj,  protein,atom_num,protein_num):
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 768]
        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        device = compound.device
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len,device)
        compound = self.gcn(compound, adj)
        # compound = [batch size,atom_num, atom_dim]
        compound = F.relu(self.fc_2(compound))
        # compound = [batch, compound len, 768]
        enc_src, src_mask = self.encoder(protein, protein_mask)
        # enc_src = [protein len,batch , hid dim]
        out = self.decoder(compound, enc_src, compound_mask, src_mask)
        # out = [batch size, 2]
        return out

    def __call__(self, data):

        compound, adj, protein, atom_num, protein_num = data

        predicted_interaction = self.forward(compound, adj, protein, atom_num, protein_num)
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_scores = ys[:, 1]

        return  predicted_scores


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