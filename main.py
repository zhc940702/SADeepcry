from sklearn.preprocessing import OneHotEncoder
import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
import csv
from Xuan_code.final.mac_network_2022 import Protein_Crystallization
from sklearn import metrics
import math

protein_char = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'U']
MAX_SEQ_Protein = 1000

def trans_drug(x):
	temp = list(x)
	temp = [i if i in protein_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_Protein:
		temp = temp + ['0'] * (MAX_SEQ_Protein-len(temp))
	else:
		temp = temp [:MAX_SEQ_Protein]
	return temp

enc_drug = OneHotEncoder().fit(np.array(protein_char).reshape(-1, 1))

def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray()

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        row = map(float, row)
        row = list(row)
        SaveList.append(row)
    return

def read_fasta(input):
    with open(input,'r') as f:
        fasta = {} 
        for line in f:
            line = line.strip()
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                fasta[header] = fasta.get(header,'') + sequence
    return fasta

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9139, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 9139),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def fold_files(args):
    rawdata_dir = args.rawpath
    fa = read_fasta(rawdata_dir + '/' + 'TE_Sequence.fasta')
    test_protein_sequences = [i for i in fa.values()]
    test_names = [i for i in fa.keys()]

    split_protein_test = []
    # total = 0
    for i in range(len(test_protein_sequences)):
        split_protein_test.append(drug_2_embed(trans_drug(test_protein_sequences[i])))

    test_protein_label = []
    test_label_file = rawdata_dir + '/' + 'TE_label'
    with open(test_label_file) as file_obj:
        for row in file_obj:
            test_protein_label.append(int(row))

    test_protein_label = np.array(test_protein_label)

    List_test_mask = []
    ReadMyCsv(List_test_mask, rawdata_dir + '/' + 'TE_feature.csv')
    List_test_mask = np.array(List_test_mask)

    fa = read_fasta(rawdata_dir + '/' + 'TR_Sequence.fasta')

    train_protein_sequences = [i for i in fa.values()]

    split_protein_train = []
    # total = 0
    for i in range(len(train_protein_sequences)):
        split_protein_train.append(drug_2_embed(trans_drug(train_protein_sequences[i])))

    train_protein_label = []
    train_label_file = rawdata_dir + '/' + 'TR_Label'
    with open(train_label_file) as file_obj:
        for row in file_obj:
            train_protein_label.append(int(row))

    train_protein_label = np.array(train_protein_label)
    List_train_mask = []
    ReadMyCsv(List_train_mask, rawdata_dir + '/' + 'TR_feature.csv')
    List_train_mask = np.array(List_train_mask)
    return split_protein_test, List_test_mask, test_protein_label, split_protein_train, List_train_mask, train_protein_label

def Generate_results(args):
    test_protein_sequences, List_test, test_protein_label, train_protein_sequences, List_train, train_protein_label = fold_files(args)
    data_test = np.zeros((len(test_protein_sequences), 1))
    for i in range(data_test.shape[0]):
        data_test[i] = i

    testset = torch.utils.data.TensorDataset(torch.LongTensor(data_test), torch.LongTensor(test_protein_sequences), torch.FloatTensor(List_test),
                                             torch.LongTensor(test_protein_label))
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                        num_workers=0, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    config = {'max_protein_seq': 1000,
              'emb_size': 100,
              'dropout_rate': 0.5,
              'input_dim_target': 21,
              'hidden_size': 100,
              'num_attention_heads': 10,
              'attention_probs_dropout_prob': 0.5,
              'intermediate_size': 100,
              'hidden_dropout_prob': 0.5
              }
    model = Protein_Crystallization(21, 256, args.embed_dim, 4, args.droprate, **config).to(device)
    vae = VAE().to(device)

    rawdata_dir = args.rawpath

    vae.load_state_dict(torch.load(rawdata_dir + '/' + 'vae.pkl'))
    model.load_state_dict(torch.load(rawdata_dir + '/' + 'model.pkl'))
    final_prediction = Generate_predictions(args, model, vae, _test, device)
    rawdata_dir = args.rawpath
    fa = read_fasta(rawdata_dir + '/' + 'TE_Sequence.fasta')
    test_names = [i for i in fa.keys()]
    prediction = []
    for i in range(len(test_names)):
        prediction.append([test_names[i], final_prediction[i]])

    print(prediction)

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims = True)
    s = x_exp / x_sum
    return s

def Generate_predictions(args, model, vae, test_loader, device):
    model.eval()
    pred1 = []
    pred3 = []
    label_truth = []
    ground_sequences = []
    ground_list = []

    for index, test_sequences, test_list, test_labels in test_loader:

        ground_sequences.append(list(test_sequences.data.cpu().numpy()))
        # ground_mask.append(list(test_mask.data.cpu().numpy()))
        ground_list.append(list(index.data.cpu().numpy()))

        test_sequences, test_list = test_sequences.to(device), test_list.to(device)

        predx, predy = vae(test_list.to(device))

        scores_one = model(test_sequences, predy, device)

        # _, predicted = torch.max(scores_one.data, 1)
        pred1.append(list(scores_one.data.cpu().numpy()[:, 1]))
        pred3.append(scores_one.data.cpu().numpy())

        label_truth.append(list(test_labels.data.cpu().numpy()))

    pred3 = np.vstack(pred3)
    for i in range(pred3.shape[0]):
        pred3[i] = softmax(pred3[i])

    final_pred = np.argmax(pred3, axis=1)


    return final_pred
def main():
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--droprate', type = float, default=0.5,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default=20,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default=20,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--rawpath', type=str, default='/Users/zhaohaochen/Desktop/code/Xuan_code_v1/BD_CRYS/CF_DS',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('batchsize: ' + str(args.batch_size))
    Generate_results(args)

if __name__ == "__main__":
    main()
