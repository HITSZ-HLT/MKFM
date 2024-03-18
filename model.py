import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelWithLMHead
from model_utils import *


class BertERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)
        # bert_encoder
        self.bert_config = BertConfig.from_json_file(args.bert_model_dir + 'config.json')

        self.bert = BertModel.from_pretrained(args.home_dir + args.bert_model_dir, config = self.bert_config)
        in_dim =  args.bert_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers- 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, token_types,utterance_len,seq_len):

        # the embeddings for bert
        # if len(content_ids)>512:
        #     print('ll')

        #
        ## w token_type_ids
        # lastHidden = self.bert(content_ids, token_type_ids = token_types)[1] #(N , D)
        ## w/t token_type_ids
        lastHidden = self.bert(content_ids)[1] #(N , D)

        final_feature = self.dropout(lastHidden)

        # pooling

        outputs = self.out_mlp(final_feature) #(N, D)

        return outputs


class DAGERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_emb = nn.Embedding(2,args.hidden_dim)
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        else:
            gats = []
            for _ in range(args.gnn_layers):
                gats += [Gatdot(args.hidden_dim) if args.no_rel_attn else Gatdot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus = []
        for _ in range(args.gnn_layers):
            grus += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus = nn.ModuleList(grus)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        if self.rel_attn:
            rel_ft = self.rel_emb(s_mask) # (B, N, N, D)

        H0 = F.relu(self.fc1(features)) # (B, N, D)
        H = [H0]
        for l in range(self.args.gnn_layers):
            H1 = self.grus[l](H[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            for i in range(1, num_utter):
                if not self.rel_attn:
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                else:
                    _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], rel_ft[:, i, :i, :])
                H1 = torch.cat((H1 , self.grus[l](H[l][:,i,:], M).unsqueeze(1)), dim = 1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
            H0 = H1
        H.append(features)
        H = torch.cat(H, dim = 2) #(B, N, l*D)
        logits = self.out_mlp(H)
        return logits

# IEMOCAP
class DAGERC_fushion(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                # gats += [GAT_dialoggcn(args.hidden_dim)]
                gats += [GAT_dialoggcn_v1_2(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)

        ### for CAGAT ########################################
        # if self.args.attn_type == 'linear':
        #     xgats = []
        #     for _ in range(args.gnn_layers):
        #         xgats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
        #     self.xgather = nn.ModuleList(xgats)
        # elif self.args.attn_type == 'dotprod':
        #     xgats = []
        #     for _ in range(args.gnn_layers):
        #         xgats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
        #     self.xgather = nn.ModuleList(xgats)
        # elif self.args.attn_type == 'rgcn':
        #     xgats = []
        #     for _ in range(args.gnn_layers):
        #         xgats += [GAT_dialoggcn_v1(args.hidden_dim)]
        #     self.xgather = nn.ModuleList(xgats)

        # if self.args.attn_type == 'linear':
        #     ogats = []
        #     for _ in range(args.gnn_layers):
        #         ogats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
        #     self.gather = nn.ModuleList(ogats)
        # elif self.args.attn_type == 'dotprod':
        #     ogats = []
        #     for _ in range(args.gnn_layers):
        #         ogats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
        #     self.ogather = nn.ModuleList(ogats)
        # elif self.args.attn_type == 'rgcn':
        #     ogats = []
        #     for _ in range(args.gnn_layers):
        #         ogats += [GAT_dialoggcn_v1(args.hidden_dim)]
        #     self.ogather = nn.ModuleList(ogats)

        # if self.args.attn_type == 'linear':
        #     xxgats = []
        #     for _ in range(args.gnn_layers):
        #         xxgats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
        #     self.xxgather = nn.ModuleList(xxgats)
        # elif self.args.attn_type == 'dotprod':
        #     xxgats = []
        #     for _ in range(args.gnn_layers):
        #         xxgats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
        #     self.xxgather = nn.ModuleList(xxgats)
        # elif self.args.attn_type == 'rgcn':
        #     xxgats = []
        #     for _ in range(args.gnn_layers):
        #         # gats += [GAT_dialoggcn(args.hidden_dim)]
        #         xxgats += [GAT_dialoggcn_v1(args.hidden_dim)]
        #     self.xxgather = nn.ModuleList(xxgats)

        # grus_xc = []
        # for _ in range(args.gnn_layers):
        #     grus_xc += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_xc = nn.ModuleList(grus_xc)

        # grus_xp = []
        # for _ in range(args.gnn_layers):
        #     grus_xp += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_xp = nn.ModuleList(grus_xp)

        # grus_oc = []
        # for _ in range(args.gnn_layers):
        #     grus_oc += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_oc = nn.ModuleList(grus_oc)

        # grus_op = []
        # for _ in range(args.gnn_layers):
        #     grus_op += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_op = nn.ModuleList(grus_op)

        # grus_xxc = []
        # for _ in range(args.gnn_layers):
        #     grus_xxc += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_xxc = nn.ModuleList(grus_xxc)

        # grus_xxp = []
        # for _ in range(args.gnn_layers):
        #     grus_xxp += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        # self.grus_xxp = nn.ModuleList(grus_xxp)
        
        # xfcs = []
        # for _ in range(args.gnn_layers):
        #     xfcs += [nn.Linear(args.hidden_dim, args.hidden_dim)]
        # self.xfcs = nn.ModuleList(xfcs)

        # ofcs = []
        # for _ in range(args.gnn_layers):
        #     ofcs += [nn.Linear(args.hidden_dim, args.hidden_dim)]
        # self.ofcs = nn.ModuleList(ofcs)

        # xxfcs = []
        # for _ in range(args.gnn_layers):
        #     xxfcs += [nn.Linear(args.hidden_dim, args.hidden_dim)]
        # self.xxfcs = nn.ModuleList(xxfcs)

        ######################################################

        # fcs = []
        # for _ in range(args.gnn_layers):
        #     fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        # self.fcs = nn.ModuleList(fcs)

        self.fc1 = nn.Linear(args.emb_dim * 4, args.hidden_dim)

        ### for CAGAT ########################################
        # self.xfc = nn.Linear(768 * 3, args.hidden_dim)
        # self.ofc = nn.Linear(768 * 3, args.hidden_dim)
        # self.xxfc = nn.Linear(768 * 3, args.hidden_dim)
        ######################################################

        self.nodal_att_type = args.nodal_att_type
        
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim * 4

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features, fffeatures, adj, aadj, s_mask,s_mask_onehot, lengths, xWant, xEffect, xReact, oWant, oEffect, oReact, xIntent, xAttr, xNeed, adj2, sssmask, uttr, args):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        # print(features.shape, features2.shape)
        [features2, features3, features4] = fffeatures
        [s_mask2, s_mask3, s_mask4] = sssmask
    # parser.add_argument('--EC', action='store_true', default=False, help='does not use emtional cause')
    # parser.add_argument('--CS', action='store_true', default=False, help='does not use common')
    # parser.add_argument('--ACS', action='store_true', default=False, help='does not use affective common')
    # acom com ec

        if not args.EC:
            features4 = torch.zeros_like(features4)
        if not args.CS:
            features3 = torch.zeros_like(features3)
        if not args.ACS:
            features2 = torch.zeros_like(features2)

        features = torch.cat((features, features2, features3, features4), dim=2)

        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        # H0 = self.dropout(H0)
        H = [H0]

        ### CAGAT 1 #########################################
        # xf = torch.cat((xWant, xEffect, xReact), dim = 2)
        # of = torch.cat((oWant, oEffect, oReact), dim = 2)
        # xxf = torch.cat((xIntent, xAttr, xNeed), dim = 2)

        # xf0 = F.relu(self.xfc(xf))
        # of0 = F.relu(self.ofc(of))
        # xxf0 = F.relu(self.xxfc(xxf))

        # xH = [xf0]
        # oH = [of0]
        # xxH = [xxf0]
        ####################################################

        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1)
            M = torch.zeros_like(C).squeeze(1)
            # P = M.unsqueeze(1)
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1) 
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P

            ### CAGAT 2 #########################################
            # xH[l] = F.relu(self.xfcs[l](xH[l]))
            # xH1 = xH[l][:,0,:].unsqueeze(1)  

            # oH[l] = F.relu(self.ofcs[l](oH[l]))
            # oH1 = oH[l][:,0,:].unsqueeze(1)  

            # xxH[l] = F.relu(self.xxfcs[l](xxH[l]))
            # xxH1 = xxH[l][:,0,:].unsqueeze(1)  
            ####################################################

            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    # :param Q: (B, D) # query utterance
                    # :param K: (B, N, D) # context
                    # :param V: (B, N, D) # context
                    #print(xH1, xH)
                    att1, att2, att3, att4, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:, i, :i], aadj[:, i, :i], adj2[:,i,:i], s_mask[:,i,:i], [s_mask2[:,i,:i], s_mask3[:,i,:i], s_mask4[:,i,:i]], args)

                else:
                    print('no rgcn')
                    # if not self.rel_attn:
                    #     _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                    # else:
                    #     _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:, i, :i])

                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P

                ### CAGAT 3 #########################################
                ####################################################

                H1 = torch.cat((H1 , H_temp), dim = 1) 

                ### CAGAT 4 #########################################
                # xH1 = torch.cat((xH1 , xH[l][:,i,:].unsqueeze(1)), dim = 1) 
                # oH1 = torch.cat((oH1 , oH[l][:,i,:].unsqueeze(1)), dim = 1)
                # xxH1 = torch.cat((xxH1 , xxH[l][:,i,:].unsqueeze(1)), dim = 1)
                ####################################################

            H.append(H1)

            ### CAGAT 5 #########################################
            # xH.append(xH1)
            # oH.append(oH1)
            # xxH.append(xxH1)
            ####################################################

        H.append(features)
        
        H = torch.cat(H, dim = 2) 

        H, Hx = self.attentive_node_features(H, lengths,self.nodal_att_type) 

        logits = self.out_mlp(H)

        return logits, Hx


class DAGERC_v2(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers
  
        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        else:
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        in_dim = args.hidden_dim * (args.gnn_layers * 2 + 1) + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        if self.rel_attn:
            rel_ft = self.rel_emb(s_mask) # (B, N, N, D)

        H0 = F.relu(self.fc1(features)) # (B, N, D)
        H = [H0]
        C = [H0]
        for l in range(self.args.gnn_layers):
            CL = self.grus_c[l](C[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            M = torch.zeros_like(CL).squeeze(1)
            # P = M.unsqueeze(1)
            P = self.grus_p[l](M, C[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            for i in range(1, num_utter):
                if not self.rel_attn:
                    _, M = self.gather[l](C[l][:,i,:], P, P, adj[:,i,:i])
                else:
                    _, M = self.gather[l](C[l][:, i, :], P, P, adj[:, i, :i], rel_ft[:, i, :i, :])

                C_ = self.grus_c[l](C[l][:,i,:], M).unsqueeze(1)# (B, 1, D)
                P_ = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)# (B, 1, D)
                # P = M.unsqueeze(1)
                CL = torch.cat((CL, C_), dim = 1) # (B, i, D)
                P = torch.cat((P, P_), dim = 1) # (B, i, D)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            C.append(CL)
            H.append(CL)
            H.append(P)
        H.append(features)
        H = torch.cat(H, dim = 2) #(B, N, l*D)
        logits = self.out_mlp(H)
        return logits
