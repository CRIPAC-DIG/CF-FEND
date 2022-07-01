import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utility.wrapper import *
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from utility.norm import build_sim, build_knn_normalized_graph, build_sim_symm
from transformers import BertConfig, BertModel, BertTokenizerFast, AutoModel

DEVICE = torch.device("cuda")

class Bert_Model(nn.Module):

    def __init__(self, args, max_len=200):
        super(Bert_Model, self).__init__()
        self.config = args.config
        self.bert_cache = args.bert_cache
        self.label_num = args.label_num
        self._use_cuda = args.cuda
        self.max_len = max_len
        self.hidden_size = 768
        self.evd_size = 768

        # self.tokenizer = BertTokenizerFast.from_pretrained(self.config)
        self.bert = BertModel.from_pretrained(self.config, cache_dir=self.bert_cache)

        self.predictor = nn.Linear(2*self.hidden_size, self.label_num)
        self.attn_score = nn.Linear(768, 1)
        self.avg = torch.randn(self.evd_size).cuda()
        

    def forward(self,claim_input_id, claim_attn_mask, snippet_input_id, snippet_token_type_id, snippet_attn_mask, debias=0.0, evd_output=False):
        batch_size = claim_input_id.shape[0]

        # claim_input_id, claim_attn_mask = self.encode_claims(claims)
        claim_cls = self.bert(claim_input_id, attention_mask=claim_attn_mask)[0][:, 0, :]

        # snippet_input_id, snippet_token_type_id, snippet_attention_mask = self.encode_snippets_with_claims(snippets, claims)
        # bert can only input 2-D tensor
        snippet_input_id = snippet_input_id.view(batch_size*10, -1)
        snippet_token_type_id = snippet_token_type_id.view(batch_size*10, -1)
        snippet_attn_mask = snippet_attn_mask.view(batch_size*10, -1)
        snippet_cls = self.bert(snippet_input_id, token_type_ids=snippet_token_type_id, attention_mask=snippet_attn_mask)[0][:,0,:]
        snippet_cls = snippet_cls.view(len(claim_cls), 10, 768)

        tmp = self.attn_score(snippet_cls)
        attn_weights = torch.softmax(tmp, dim=1)
        snippet_cls *= attn_weights
        snippet_cls = torch.sum(snippet_cls, dim=1)

        claim_snippet_cls = torch.cat((claim_cls, snippet_cls), dim=-1)

        if evd_output is True:
            return snippet_cls

        if debias != 0.0:
            if self.avg is None: raise Exception("Model.avg is unavailable") 
            claim_snippet_cls_debias = torch.cat((claim_cls, self.avg.unsqueeze(0).expand(claim_cls.size(0),claim_cls.size(1))), dim=-1)
            return torch.softmax(self.predictor(claim_snippet_cls), -1) - \
             debias*torch.softmax(self.predictor(claim_snippet_cls_debias), -1)
        else:
            return self.predictor(claim_snippet_cls)

    def encode_claims(self, claims):
        tmp = self.tokenizer(claims, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        input_ids = tmp["input_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)

        return input_ids, attention_mask

    def encode_snippets_with_claims(self, snippets, claims):
        concat_claims = []
        for claim in claims:
            concat_claims += [claim]*10             # [claim1, claim1,...,claim1,claim2,..., claim_n]

        concat_snippets = [item for sublist in snippets for item in sublist.tolist()]

        tmp = self.tokenizer(concat_claims, concat_snippets, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)

        input_ids = tmp["input_ids"].to(DEVICE)
        token_type_ids = tmp["token_type_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)
        return input_ids, token_type_ids, attention_mask

    def set_avg(self, avg: torch.tensor = None):
        if avg is not None:
            self.avg = avg
        else:
            self.avg = torch.randn(self.evd_size).cuda()

class MAC(nn.Module):

    def __init__(self, args, extra_params):
        super(MAC, self).__init__()
        self.config = args.config
        self.bert_cache = args.bert_cache
        self.label_num = args.label_num
        self.hidden_size = args.hidden_size
        self.claim_len = args.claim_length
        self.evd_len = args.snippet_length
        self.output_size = args.label_num
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout
        self.evd_size = 2*self.hidden_size
        self.num_att_heads_for_words = args.num_att_heads_for_words
        self.num_att_heads_for_evds = args.num_att_heads_for_evds
        self.use_claim_source = False

        self.embedding_type = args.embedding
        if self.embedding_type == 'bert':
            self.embedding_size = 768
            self.bert = BertModel.from_pretrained(self.config, cache_dir=self.bert_cache)
            for param in self.bert.parameters():
                param.requires_grad = False

        elif self.embedding_type == 'glove':
            self.embedding_size = 300
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(extra_params['embedding_matrix'], dtype=torch.float32))

        self.claim_lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            dropout=float(self.dropout),
                            bidirectional=True,
                            batch_first=True)
        self.evd_lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            dropout=float(self.dropout),
                            bidirectional=True,
                            batch_first=True)
    
        # use claim source or evdience source
        self.use_claim_source = args.use_claim_source
        self.use_evd_source = args.use_evd_source
        self.claim_emb_size = 0
        self.evd_emb_size =0

        dim = 2*self.hidden_size
        self._get_word_attention_func(dim=dim)
        self._get_evd_attention_func(dim=dim)

        self.claim_input_size = dim  # the first is for claim, the second is for
        if self.use_claim_source: 
            self.claim_input_size += self.claim_emb_size
        self.evd_input_size = dim * self.num_att_heads_for_words * self.num_att_heads_for_evds  # twice times for two times attention
        if self.use_evd_source: 
            self.evd_input_size += self.article_emb_size * self.num_att_heads_for_evds
        self.out = nn.Sequential(
            nn.Linear(self.claim_input_size+self.evd_input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.out[0].apply(init_weights)
        self.out[1].apply(init_weights)

        self.avg = torch.randn(self.evd_input_size).cuda()

    def forward(self, claim_id, claim_mask, evidence_id, _, evidence_mask, debias=0.0, evd_output=False):
        B = claim_id.shape[0]
         
        if self.embedding_type == 'bert':
            claim = self.bert(claim_id, attention_mask=claim_mask)[0] # (batch, len, dim)
            evidence = self.bert(evidence_id.view(B*10, -1), attention_mask=evidence_mask.view(B*10, -1))[0] 
        elif self.embedding_type == 'glove':
            claim = self.embedding(claim_id)
            evidence = self.embedding(evidence_id.view(B*10, -1))

        claim = self._generate_query_repr(claim, claim_mask)           # (batch, 2*self.hidden_size)
        
        evd_length = torch.sum(evidence_mask.view(B*10, -1),dim=1)
        pack_padded_evidence = pack_padded_sequence(evidence, evd_length.cpu(), batch_first=True, enforce_sorted=False)
        bilstm_evidence, _ = self.evd_lstm(pack_padded_evidence)              
        bilstm_evidence, _ = pad_packed_sequence(bilstm_evidence, batch_first=True, total_length=self.evd_len)            # (batch*10, len, 2*hidden_size)

        avg, word_att_weights = self._word_level_attention(left_tsr=claim.unsqueeze(1).expand(B, 10, -1).reshape(B*10, -1), right_tsr=bilstm_evidence,
                                                           right_mask=evidence_mask.view(B*10, -1),)   # avg: (B*10, heads_for_words*hidden_size)
        avg, evd_att_weight = self._evidence_level_attention_new(claim, avg.view(B, 10, -1), evidence_mask)            # avg: (B, heads_for_words*head_for_evds*hidden_size)   
        output = torch.cat([claim, avg], dim=-1)         
        
        if evd_output is True:
            return avg

        if debias != 0.0:
            if self.avg is None: raise Exception("Model.avg is unavailable") 
            output2 = torch.cat((claim.detach().clone(), self.avg.unsqueeze(0).expand(B, -1)), dim=-1)
            return torch.softmax(self.out(output),-1) - debias* torch.softmax(self.out(output2),-1)
        else:
            return self.out(output)
        

    def _generate_query_repr(self, claim: torch.Tensor, claim_mask: torch.Tensor):
        claim_length = torch.sum(claim_mask, dim=1)        # (batch,) 
        pack_padded_claim = pack_padded_sequence(claim, claim_length.cpu(), batch_first=True,enforce_sorted=False)   # enforced_sorted: sort
        bilstm_claim, _ = self.claim_lstm(pack_padded_claim)                            
        bilstm_claim, _ = pad_packed_sequence(bilstm_claim, batch_first=True, total_length=self.claim_len)      # (batch, len, 2*hidden_size)
        
        claim_length = claim_length.unsqueeze(1)
        claim_repr = torch.sum(bilstm_claim * claim_mask.unsqueeze(2).float(), dim=1) / claim_length     # (batch, dim)       
        return claim_repr                    
                        
    def _word_level_attention(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, right_mask: torch.Tensor, **kargs):
        """
            Compute word-level attention of evidences.
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, H). It represents claims' representation
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, R, H). Doc's representations.
        right_mask: `torch.Tensor` (n1 + n2 + ... + nx, R)
        kargs
        Returns
        -------
            Representations of each of evidences of each of claim in the mini-batch of shape (B1, X)
        """
        # for reproducing results in the report
        B1, R, H = right_tsr.size() # [batch*10, len, 2*self.hidden_size]a
        assert left_tsr.size(0) == B1 and len(left_tsr.size()) == 2
        # new_left_tsr = left_tsr.unsqueeze(1).expand(B1, R, -1)
        avg, att_weight = self.self_att_word(left_tsr, right_tsr, right_mask)
        avg = torch.flatten(avg, start_dim=1)  # (n1 + n2 + n3 + ... + nx, n_head * 4D)
        # avg = torch.cat([left_tsr, avg], dim=-1)  # (B1, 2D + D)
        return avg, att_weight  # (n1 + n2 + n3 + ... + nx, R)

    def _evidence_level_attention_new(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor,
                                      evidence_mask: torch.Tensor,):
        """
        compute evidence-level attention
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (B, D)
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, D)
        full_padded_document: `torch.Tensor` (B, R). Note, B != (n1 + n2 + ... + nx)

        Returns
        -------
            a tensor of shape (B, _) which stands for representation of `batch_size = B` claims in each of mini-batches
        """
        # for reproducing results in the report
        # if self.evd_attention_type != AttentionType.ConcatNotEqual: left_tsr = self.map_query_level2(left_tsr)
        
        mask = (torch.sum(evidence_mask, dim=-1) > 4).float()  # (B, n), 0 is for padding; min evidence length is 4
        # if self.use_article_source:
        #     right_tsr = self._use_article_embeddings(right_tsr, **kargs)

        attended_avg, att_weight = self.self_att_evd(left_tsr, right_tsr, mask)
        avg = torch.flatten(attended_avg, start_dim=1)  # (B, num_heads * 2D)
        return avg, att_weight


    def _get_word_attention_func(self, dim: int):
        """
        get the function to compute attention weights on word.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        input_dim = 2 * dim
        self.self_att_word = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim,
                                                    num_heads=self.num_att_heads_for_words)
        # else:
        #     raise NotImplemented("Unknown attention type for words")`  

    def _get_evd_attention_func(self, dim: int):
        """
        get the function to compute attention weights on evidence.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        # the first is for claim, the second is for word att on evds
        input_dim = dim + self.num_att_heads_for_words * dim
        # if self.use_claim_source: input_dim += self.claim_emb_size
        # if self.use_article_source: input_dim += self.article_emb_size
        self.self_att_evd = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim, num_heads=self.num_att_heads_for_evds)
        # else:
        #     raise NotImplemented("Unknown attention type for evidences")  

    def set_avg(self, avg: torch.tensor = None):
        if avg is not None:
            self.avg = avg
        else:
            self.avg = torch.randn(self.evd_input_size).cuda()
    

class ConcatNotEqualSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor):
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


def init_weights(m):
    """
    Copied from https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/3
    Examples:
        >>> w = nn.Linear(3, 4)
        >>> w.apply(init_weights)
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m.bias, "data"): m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)

