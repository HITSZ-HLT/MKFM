import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from utils import person_embed
from tqdm import tqdm
import json

def create_class_weight_SCL(label):
    unique = [0, 1]
    one = sum(label)
    labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(total/labels_dict[key])
        weights.append(score)
    return weights

def train_or_eval_model(ss, model, loss_function, stSCL, dataloader, epoch, cuda, args, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    assert not train or optimizer != None
    if train:
        model.train()
        # dataloader = tqdm(dataloader)
    else:
        model.eval()

    cnt = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
            
        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        features, [features2, features3, features4], label, adj, aadj, s_mask, s_mask_onehot,lengths, speakers, utterances, xWant, xEffect, xReact, oWant, oEffect, oReact, xIntent, xAttr, xNeed, adj2, [s_mask2, s_mask3, s_mask4], [ll1, ll2, ll3] = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            features, features2, features3, features4 = features.cuda(), features2.cuda(), features3.cuda(), features4.cuda()
            label = label.cuda()
            speakers, ll1, ll2, ll3 = speakers.cuda(), ll1.cuda(), ll2.cuda(), ll3.cuda()
            adj, aadj, adj2 = adj.cuda(), aadj.cuda(), adj2.cuda()
            s_mask, s_mask2, s_mask3, s_mask4 = s_mask.cuda(), s_mask2.cuda(), s_mask3.cuda(), s_mask4.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            lengths = lengths.cuda()
            xWant, xEffect, xReact = xWant.cuda(), xEffect.cuda(), xReact.cuda()
            oWant, oEffect, oReact = oWant.cuda(), oEffect.cuda(), oReact.cuda()
            xIntent, xAttr, xNeed = xIntent.cuda(), xAttr.cuda(), xNeed.cuda()

        log_prob, H = model(features, [features2, features3, features4], adj, aadj, s_mask, s_mask_onehot, lengths, xWant, xEffect, xReact, oWant, oEffect, oReact, xIntent, xAttr, xNeed, adj2, [s_mask2, s_mask3, s_mask4], utterances, args) # (B, N, C)
        
        loss = loss_function(log_prob.permute(0,2,1), label)

        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist()
        preds += pred
        labels += label
        losses.append(loss.item())

        ######## SsCL ##########################################################################
        dim = H.shape[2]
        # [scarcasm, mp, topic]
        
        [ttt, sss, ppp] = ss

        tpw = 0
        if args.TP:
            tpw = ttt
        scw = 0
        if args.SC:
            scw = sss
        mpw = 0
        if args.MP:
            mpw = ppp

        loss_1 = stSCL(H.view(-1, dim), ll1.view(-1))
        loss_2 = stSCL(H.view(-1, dim), ll2.view(-1))
        loss_3 = stSCL(H.view(-1, dim), ll3.view(-1))
        ########################################################################################

        if train:
            loss_val = loss.item()
            loss = loss + scw * loss_1 + mpw * loss_2 + tpw * loss_3

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        new_preds_w_es = []
        new_preds_wo_es = []
        new_labels_w_es = []
        new_labels_wo_es = []

        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    if j == 0:
                        new_labels_wo_es.append(l)
                        new_preds_wo_es.append(preds[i][j])
                    else:
                        if label[j] == label[j - 1]:
                            new_labels_wo_es.append(l)
                            new_preds_wo_es.append(preds[i][j])
                        else:
                            new_labels_w_es.append(l)
                            new_preds_w_es.append(preds[i][j])
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan')#, [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        avg_fscore_w_es = round(f1_score(new_labels_w_es, new_preds_w_es, average='weighted') * 100, 2)
        avg_fscore_wo_es = round(f1_score(new_labels_wo_es, new_preds_wo_es, average='weighted') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_fscore#, avg_fscore_w_es, avg_fscore_wo_es
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        return avg_loss, avg_accuracy, labels, preds, avg_micro_fscore, avg_macro_fscore


def save_badcase(model,  dataloader, cuda, args, speaker_vocab, label_vocab):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []

    model.eval()

    for data in dataloader:

        # text_ids, text_feature, speaker_ids, labels, umask = [d.cuda() for d in data] if cuda else data
        features, label, adj,s_mask, s_mask_onehot,lengths, speaker, utterances = data
        # speaker_vec = person_embed(speaker_ids, person_vec)
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()


        # print(speakers)
        log_prob = model(features, adj,s_mask, s_mask_onehot, lengths) # (B, N, C)


        label = label.cpu().numpy().tolist() # (B, N)
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker

        # finished here

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return

    cases = []
    for i,d in enumerate(dialogs):
        case = []
        for j,u in enumerate(d):
            case.append({
                'text': u,
                'speaker': speaker_vocab['itos'][speakers[i][j]],
                'label': label_vocab['itos'][labels[i][j]] if labels[i][j] != -1 else 'none',
                'pred': label_vocab['itos'][preds[i][j]]
            })
        cases.append(case)

    with open('badcase/%s.json'%(args.dataset_name), 'w', encoding='utf-8') as f:
        json.dump(cases,f)

    # print(preds.tolist())
    # print(labels.tolist())
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
        print('badcase saved')
        print('test_f1', avg_fscore)
        return
    else:
        avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
        avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
        print('badcase saved')
        print('test_micro_f1', avg_micro_fscore)
        print('test_macro_f1', avg_macro_fscore)
        return
