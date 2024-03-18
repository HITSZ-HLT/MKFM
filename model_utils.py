import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) \
                   / torch.sum(self.weight[target])
        return loss

class GatedSelection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.context_trans = nn.Linear(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x2 = self.context_trans(x2)
        s = self.sigmoid(self.linear1(x1)+self.linear2(x2))
        h = s * x1 + (1 - s) * x2
        return self.relu(self.fc(h))

def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30

class GatLinear(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, 1)


    def forward(self, Q, K, V, adj):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :return:
        '''
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

class GatDot(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q, K, V, adj):
        '''
        imformation gatherer with dot product attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :return:
        '''
        N = K.size()[1]


        Q = self.linear1(Q).unsqueeze(2) # (B,D,1)
        # K = self.linear2(Q) # (B, N, D)
        K = self.linear2(K) # (B, N, D)

        alpha = torch.bmm(K, Q).permute(0, 2, 1)  # (B, 1, N)

        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj)  # (B, 1, N)

        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1)  # (B,  D)

        return attn_weight, attn_sum

class GatLinear_rel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 3, 1)
        self.rel_emb = nn.Embedding(2, hidden_size)


    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        rel_emb = self.rel_emb(s_mask) # (B, N, D)
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)
        # print('K',K.size())
        # print('rel_emb', rel_emb.size())
        X = torch.cat((Q,K, rel_emb), dim = 2) # (B, N, 2D)?   (B, N, 3D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum


class GatDot_rel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.rel_emb = nn.Embedding(2, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with dot product attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #  relation mask
        :return:
        '''
        N = K.size()[1]

        rel_emb = self.rel_emb(s_mask)
        Q = self.linear1(Q).unsqueeze(2) # (B,D,1)
        K = self.linear2(K) # (B, N, D)
        y = self.linear3(rel_emb) # (B, N, 1)

        alpha = (torch.bmm(K, Q) + y).permute(0, 2, 1)  # (B, 1, N)

        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj)  # (B, 1, N)

        attn_weight = F.softmax(alpha, dim=2)  # (B, 1, N)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1)  # (B,  D)

        return attn_weight, attn_sum


class GAT_dialoggcn(nn.Module):
    '''
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        self.rel_emb = nn.Parameter(torch.randn(2, hidden_size, hidden_size))

    def forward(self, Q, K, V, adj, s_mask_onehot):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N, 2) #
        :return:
        '''
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        # print('s_mask_onehot', s_mask_onehot.size())
        D = self.rel_emb.size()[2]
        # print('rel_emb', self.rel_emb.size())
        rel_emb = self.rel_emb.unsqueeze(0).expand(B,-1,-1,-1)
        # rel_emb = self.rel_emb.unsqueeze(0).repeat(B, 1, 1, 1)
        # print('rel_emb expand', rel_emb.size())

        rel_emb = rel_emb.reshape((B, 2, D*D))
        # print('rel_emb resize', rel_emb.size())
        Wr = torch.bmm(s_mask_onehot, rel_emb).reshape((B, N, D, D)) # (B, N, D, D)
        # print('Wr', Wr.size()) # (B, N, D, D)

        Wr = Wr.reshape((B*N, D, D))
        # print('Wr after reshape', Wr.size())

        V = V.unsqueeze(2).reshape((B*N, 1, -1)) # (B*N, 1, D)
        # print('V after reshape', V.size())
        V = torch.bmm(V, Wr).unsqueeze(1) #(B * N,  D)
        # print('V after transform', V.size())
        V = V.reshape((B,N,-1))
        # print('Final V', V.size())

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

# IEMOCAP
class GAT_dialoggcn_v1(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)

        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        #alpha = F.leaky_relu(alpha)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()   # (B, N, 1)
        V = V0 * s_mask + V1 * (1 - s_mask)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum

class GAT_dialoggcn_v1_2(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 2, 1)
        # self.tlinear = nn.Linear(hidden_size * 2, 1)

        # self.xlinear = nn.Linear(hidden_size * 2, 1) # xWant, xEffect, xReact
        # self.olinear = nn.Linear(hidden_size * 2, 1) # oWant, oEffect, oReact
        # self.xxlinear = nn.Linear(hidden_size * 2, 1) # xIntent, xAttr, xNeed
        self.f_linear = nn.Linear(hidden_size * 4, hidden_size)
        # self.f_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.tW2 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.tW3 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.tW4 = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, Q, K, V, adj, aadj, adj2, s_mask, sssmask, args):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''

        [s_mask2, s_mask3, s_mask4] = sssmask
        
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K), dim = 2) # (B, N, 2D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        #alpha2 = self.tlinear(X).permute(0,2,1) #(B, 1, N)
        
        #alpha = F.leaky_relu(alpha)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)  # (B, 1, N)
        alpha = mask_logic(alpha, adj) # (B, 1, N)

        # topic
        #adj2 = adj2.unsqueeze(1)  # (B, 1, N)
        #alpha2 = mask_logic(alpha2, adj2) # (B, 1, N)

        # xQ = xQ.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # oQ = oQ.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # xxQ = xxQ.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # xX = torch.cat((xQ,xK), dim = 2) # (B, N, 2D)
        # oX = torch.cat((oQ,oK), dim = 2) # (B, N, 2D)
        # xxX = torch.cat((xxQ,xxK), dim = 2) # (B, N, 2D)

        # xalpha = self.xlinear(xX).permute(0,2,1) #(B, 1, N)
        # oalpha = self.olinear(oX).permute(0,2,1) #(B, 1, N)
        # xxalpha = self.xxlinear(xxX).permute(0,2,1) #(B, 1, N)
        
        # aadj = aadj.unsqueeze(1)  # (B, 1, N)
        # aalpha = mask_logic(alpha, aadj) # (B, 1, N)
        # xalpha = mask_logic(xalpha, aadj)
        # oalpha = mask_logic(oalpha, aadj)
        # xxalpha = mask_logic(xxalpha, aadj)
        #print(xalpha.shape)
        #print(alpha[0], xalpha[0], oalpha[0], xxalpha[0])

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # aattn_weight = F.softmax(aalpha, dim = 2) # (B, 1, N)
        #tattn_weight = F.softmax(alpha2, dim = 2) # (B, 1, N)

        # xattn_weight = F.softmax(xalpha, dim = 2) # (B, 1, N)
        # oattn_weight = F.softmax(oalpha, dim = 2) # (B, 1, N)
        # xxattn_weight = F.softmax(xxalpha, dim = 2) # (B, 1, N)
        
        #print(s_mask)
        #print(attn_weight[0:5], xattn_weight[0:5], oattn_weight[0:5], xxattn_weight[0:5])
        # xattn_weight = torch.mul(xattn_weight, s_mask.unsqueeze(1).float())
        # oattn_weight = torch.mul(oattn_weight, (1 - s_mask).unsqueeze(1).float())
        # xxattn_weight = torch.mul(xxattn_weight, s_mask.unsqueeze(1).float())
        #print(attn_weight[0:5], xattn_weight[0:5], oattn_weight[0:5], xxattn_weight[0:5])
        V0 = self.Wr0(V) # (B, N, D)
        V1 = self.Wr1(V) # (B, N, D)

        tV2 = self.tW2(V) # (B, N, D)
        tV3 = self.tW3(V) # (B, N, D)
        tV4 = self.tW4(V) # (B, N, D)

        s_mask  = s_mask.unsqueeze(2).float()   # (B, N, 1)
        s_mask2 = s_mask2.unsqueeze(2).float()   # (B, N, 1)
        s_mask3 = s_mask3.unsqueeze(2).float()   # (B, N, 1)
        s_mask4 = s_mask4.unsqueeze(2).float()   # (B, N, 1)

        # intra --> s_mask
        # inter --> 1 - s_mask
        V = V0 * s_mask + V1 * (1 - s_mask)
        tV2 = tV2 * s_mask2 + tV2 * (1 - s_mask2)
        tV3 = tV3 * s_mask3 + tV2 * (1 - s_mask3)
        tV4 = tV4 * s_mask4 + tV2 * (1 - s_mask4)

    # parser.add_argument('--CR', action='store_true', default=False, help='does not use co-reference')
    # parser.add_argument('--CT', action='store_true', default=False, help='does not use context')
    # parser.add_argument('--EC2', action='store_true', default=False, help='does not use emtional cause2')
    # pronoun   cause   context

        #print(attn_weight.shape, xattn_weight.shape)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        attn_sum2 = torch.bmm(attn_weight, tV2).squeeze(1) # (B, D)
        attn_sum3 = torch.bmm(attn_weight, tV3).squeeze(1) # (B, D)
        attn_sum4 = torch.bmm(attn_weight, tV4).squeeze(1) # (B, D)

        if not args.CR:
            attn_sum2 = torch.zeros_like(attn_sum2)
        if not args.CT:
            attn_sum4 = torch.zeros_like(attn_sum4)
        if not args.EC2:
            attn_sum3 = torch.zeros_like(attn_sum3)

        # aattn_sum = torch.bmm(aattn_weight, V).squeeze(1) # (B, D)
        # tattn_sum = torch.bmm(attn_weight, tV).squeeze(1) # (B, D)
        # attn_sum = attn_sum + aattn_sum
        # xattn_sum = torch.bmm(xattn_weight, V).squeeze(1) # (B, D)
        # oattn_sum = torch.bmm(oattn_weight, V).squeeze(1) # (B, D)
        # xxattn_sum = torch.bmm(xxattn_weight, V).squeeze(1) # (B, D)

        attn_sum = torch.cat((attn_sum, attn_sum2, attn_sum3, attn_sum4), dim = 1)  # (B, 2D)
        attn_sum = self.f_linear(attn_sum) #(B, 1, N)
        attn_sum_f = attn_sum# + 1 * tattn_sum

        return attn_weight, None, None, None, attn_sum_f


class GAT_dialoggcn_v2(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j, rel)
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * 3, 1)
        self.Wr0 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.Wr1 = nn.Linear(hidden_size, hidden_size, bias = False)
        self.rel_emb = nn.Embedding(2, hidden_size)

    def forward(self, Q, K, V, adj, s_mask):
        '''
        imformation gatherer with linear attention
        :param Q: (B, D) # query utterance
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B,  N) # the adj matrix of the i th node
        :param s_mask: (B,  N) #
        :return:
        '''
        rel_emb = self.rel_emb(s_mask) # (B, N, D)
        B = K.size()[0]
        N = K.size()[1]
        # print('Q',Q.size())
        Q = Q.unsqueeze(1).expand(-1, N, -1) # (B, N, D)；
        # print('K',K.size())
        X = torch.cat((Q,K,rel_emb), dim = 2) # (B, N, 3D)
        # print('X',X.size())
        alpha = self.linear(X).permute(0,2,1) #(B, 1, N)
        # print('alpha',alpha.size())
        # print(alpha)
        adj = adj.unsqueeze(1)
        alpha = mask_logic(alpha, adj) # (B, 1, N)
        # print('alpha after mask',alpha.size())
        # print(alpha)

        attn_weight = F.softmax(alpha, dim = 2) # (B, 1, N)
        # print('attn_weight',attn_weight.size())
        # print(attn_weight)

        V0 = self.Wr0(V) # (B, N,D)
        V1 = self.Wr1(V) # (B, N, D)

        s_mask = s_mask.unsqueeze(2).float()
        V = V0 * s_mask + V1 * (1 - s_mask)

        attn_sum = torch.bmm(attn_weight, V).squeeze(1) # (B, D)
        # print('attn_sum',attn_sum.size())

        return attn_weight, attn_sum


class attentive_node_features(nn.Module):
    '''
    Method to obtain attentive node features over the graph convoluted features
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self,features, lengths, nodal_att_type):
        '''
        features : (B, N, V)
        lengths : (B, )
        nodal_att_type : type of the final nodal attention
        '''

        if nodal_att_type==None:
            return features, features

        batch_size = features.size(0)
        max_seq_len = features.size(1)
        padding_mask = [l*[1]+(max_seq_len-l)*[0] for l in lengths]
        padding_mask = torch.tensor(padding_mask).to(features)    # (B, N)
        causal_mask = torch.ones(max_seq_len, max_seq_len).to(features)  # (N, N)
        causal_mask = torch.tril(causal_mask).unsqueeze(0)  # (1, N, N)

        if nodal_att_type=='global':
            mask = padding_mask.unsqueeze(1)
        elif nodal_att_type=='past':
            mask = padding_mask.unsqueeze(1)*causal_mask

        x = self.transform(features)  # (B, N, V)
        temp = torch.bmm(x, features.permute(0,2,1))
        #print(temp)
        alpha = F.softmax(torch.tanh(temp), dim=2)  # (B, N, N)
        alpha_masked = alpha*mask  # (B, N, N)
        
        alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # (B, N, 1)
        #print(alpha_sum)
        alpha = alpha_masked / alpha_sum    # (B, N, N)
        attn_pool = torch.bmm(alpha, features)  # (B, N, V)

        return attn_pool, x


