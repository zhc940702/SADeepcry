import numpy as np
import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import time
import torch.utils.data
from Network_train import Protein_Crystallization
import csv

CHARPROTSET = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '?': 21}

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        row = map(float, row)
        row = list(row)
        SaveList.append(row)
    return

MAX_SEQ_Protein = 1000

def read_fasta(input): #用def定义函数read_fasta()，并向函数传递参数用变量input接收
    with open(input,'r') as f: # 打开文件
        fasta = {} # 定义一个空的字典
        for line in f:
            line = line.strip() # 去除末尾换行符
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                fasta[header] = fasta.get(header,'') + sequence
    return fasta

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=800):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def drug_2_embed(x):
    return label_sequence(x,CHARPROTSET,MAX_SEQ_Protein)
    # return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray()

def fold_files(args):
    rawdata_dir = args.rawpath

    fa = read_fasta(rawdata_dir + '/' + 'TE_Sequence.fasta')
    test_protein_sequences = [i for i in fa.values()]

    split_protein_test = []
    # total = 0
    for i in range(len(test_protein_sequences)):
        split_protein_test.append(drug_2_embed(test_protein_sequences[i]))


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
        split_protein_train.append(drug_2_embed(train_protein_sequences[i]))


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

def train(model, train_loader, optimizer, lossfunction1, lossfunction2, device):

    model.train()

    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        _, batch_sequences, batch_list, batch_labels = data
        optimizer.zero_grad()

        logits, score_2 = model(batch_sequences.to(device), batch_list.to(device), device)

        loss1 = lossfunction1(score_2, batch_list.to(device))
        loss2 = lossfunction2(logits, batch_labels.to(device))

        loss = loss1 + loss2

        loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss += loss.item()

    return 0

def train_test(args):
    test_protein_sequences, List_test, test_protein_label, train_protein_sequences, List_train, train_protein_label = fold_files(args)
    data_test = np.zeros((len(test_protein_sequences), 1))
    data_train = np.zeros((len(train_protein_sequences), 1))
    for i in range(data_test.shape[0]):
        data_test[i] = i
    for j in range(data_train.shape[0]):
        data_train[j] = j

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(data_train), torch.LongTensor(train_protein_sequences), torch.FloatTensor(List_train),
                                              torch.LongTensor(train_protein_label))
    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    config = {'max_protein_seq': 800,  # 数据集中最大蛋白质长度
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
    loss_func = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # ====================   training    ====================
        train(model, _train, optimizer, loss_func, criterion, device)
