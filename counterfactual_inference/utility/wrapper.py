
import torch
import torch.nn as nn
import torch.nn.functional as F

class GGNN(nn.Module):
    """
    This is implementation of self-attention in ICLR 2016 Paper 
    Gated Graph Sequence Neural Networks, https://arxiv.org/abs/1511.05493
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        """
        Parameters
        -----------
        in_features
        out_features
        dropout 
        """
        super(GGNN, self).__init__()
        self.proj = Linear(in_features, out_features, bias=False)
        self.linearz0 = Linear(out_features, out_features)
        self.linearz1 = Linear(out_features, out_features)
        self.linearr0 = Linear(out_features, out_features)
        self.linearr1 = Linear(out_features, out_features)
        self.linearh0 = Linear(out_features, out_features)
        self.linearh1 = Linear(out_features, out_features)
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, adj, x):
        """
        Parameters
        -----------
        adj: normalized adj matrix
        x: node features

        Returns
        -----------
        """
        if hasattr(self, 'dropout'): 
            x = self.dropout(x)
        x = self.proj(x)
        a = adj.matmul(x)

        z0 = self.linearz0(a)
        z1 = self.linearz1(x)
        z = torch.sigmoid(z0 + z1)

        r0 = self.linearr0(a)
        r1 = self.linearr1(x)
        r = torch.sigmoid(r0 + r1)

        h0 = self.linearh0(a)
        h1 = self.linearh1(r*x)
        h = torch.tanh(h0 + h1)

        feat = h*z + x*(1-z)
    
        return feat

class GSL(nn.Module):
    def __init__(self, rate):
        """
        Parameters
        ------------
        rate: drop rate of GSL
        """
        super(GSL, self).__init__()
        self.rate = rate

    def forward(self, adj, score):
        """
        Parameters
        ------------
        adj: normalized adj matrix
        score: score for every node

        Returns
        -----------
        """
        N = adj.shape[-1]
        BATCH_SIZE = adj.shape[0]
        num_preserve_node = int(self.rate * N)
        _, indices = score.topk(num_preserve_node, 1)
        indices = torch.squeeze(indices, dim=-1)
        mask = torch.zeros([BATCH_SIZE, N, N]).cuda()
        for i in range(BATCH_SIZE):
            mask[i].index_fill_(0, indices[i], 1)
            mask[i].index_fill_(1, indices[i], 1)
        adj = adj * mask
        # feat = torch.tanh(score) * feat
        return adj

class GGNN_with_GSL(nn.Module):
    """
    combine GGNN and GSL
    """
    def __init__(self, input_dim, hidden_dim, output_dim, rate=0.6, dropout=0.2):
        super(GGNN_with_GSL, self).__init__()
        self.feat_prop1 = GGNN(input_dim, hidden_dim, dropout)
        self.word_scorer1 = GGNN(hidden_dim, 1, dropout)
        self.gsl1 = GSL(rate)
        self.feat_prop2 = GGNN(hidden_dim, hidden_dim, dropout)
        self.feat_prop3 = GGNN(hidden_dim, output_dim, dropout)
        
    def forward(self, adj, feat):
        feat = self.feat_prop1(adj, feat)
        score = self.word_scorer1(adj, feat)
        adj_refined = self.gsl1(adj, score)
        feat = self.feat_prop2(adj_refined, feat)
        feat = self.feat_prop3(adj_refined, feat)
        
        return feat

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        if hasattr(self, 'linear.bias'):
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'): x = self.dropout(x)
        x = self.linear(x)
        return x

