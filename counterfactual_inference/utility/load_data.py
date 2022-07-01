from cProfile import label
import os
import re
import json
import torch
import pickle
import random
import torchvision


import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing_extensions import OrderedDict
from sklearn.feature_extraction import image
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizerFast
from utility.keywords import UNK_TOKEN, MASK_TOKEN

columns = ['claim_id','claim', 'label'] + ["snippets_"+str(i) for i in range(10)]

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def convert_label(label: str, *label_order):
    label_order = list(label_order)
    return label_order.index(label)
    

def merge_label(label: str, *args):
    dataset = args[0]
    label_order = args[1]
    index = label_order.index(label)
    if dataset == 'snes':
        return 'false' if index < 2 else 'true'         # merge false, mostly false as false
        # return 'false' if index < 3 else 'true'         # merge false, mostly false, mixture as false
    else:
        return 'false' if index < 3 else 'true'         # merge pants on fire, false, mostly false as false

class Load_Data():

    def __init__(self, args, logger):
        self.dataset = args.dataset
        self.filter_websites = args.filter_websites
        self.config = args.config
        self.label_num = args.label_num
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config)
        self.claim_length = args.claim_length
        self.snippet_length = args.snippet_length
        self.logger = logger
        self.model_type = args.model
        self.embedding = args.embedding
        self.filter_mixture = args.filter_mixture
        
        
    def load_data(self, other_dataset=False):
        # load data from memory
        new_label_order = ['false', 'true']
        path_prefix = "../multi_fc_publicdata/" + self.dataset + "/" + self.dataset
        main_data = pd.read_csv("%s.tsv" % (path_prefix), sep='\t', header=None)
        snippets_data = pd.read_csv("%s_snippets.tsv" % (path_prefix), sep='\t', header=None )
        label_order = pickle.load(open("%s_labels.pkl" %(path_prefix), "rb")) # 1-d list
        splits = pickle.load(open("%s_index_split.pkl" % (path_prefix), "rb")) # 2-d list
        
        
        if self.label_num != len(label_order):
            # merge labels
            main_data.iloc[:, 2] = main_data.iloc[:,2].apply(merge_label, args=(self.dataset, label_order))
            label_order = new_label_order
            
        hard_splits = pickle.load(open("%s_error_split%d.pkl"%(path_prefix, len(label_order)),'rb'))   # 1-d list, item: str
        hard_splits = main_data[main_data.iloc[:, 0].isin(hard_splits)].index.tolist() # 1-d list
        splits += [hard_splits]

        main_data.iloc[:, 2] = main_data.iloc[:,2].apply(convert_label, args=(label_order))
        
        print(main_data.head())
        
        # ------------------filter snippets according to its website--------
        if self.filter_websites > 0.5:
            # don't understand why to do this
            snippets_data = self.filter_websites(snippets_data)
        
        # count the ratio of each label
        all_labels = main_data.values[:, 2]
        counter = Counter(all_labels)  # Dict subclass for counting hashable items. Elements are stored as dictionary keys and their counts are stored as dictionary values.
        data_distribution_desc = ""
        for idx, label_name in enumerate(label_order):
            data_distribution_desc = data_distribution_desc  + ", " + str(label_name) + " (" + str(counter[idx]) + ", " + str(np.around(counter[idx]/len(all_labels) * 100,1)) + "%)"
        self.logger.logging("Total labels: %d, labels distributon: %s" %(len(all_labels), data_distribution_desc))

        # extract train/val/test data
        data = {}
        for key, split in zip(['train', 'val', 'test', 'hard'], splits):
            sub_main_data = main_data.loc[split, :2]         # pandas.DataFrame, claim_id, claim, label
            sub_snippets_data = snippets_data.loc[split, 1:]

            self.logger.logging('key: {}, main_data: {}, snippets: {}'.format(key, sub_main_data.shape, sub_snippets_data.shape))
            sub_labels = sub_main_data.values[:, 2]

            counter = Counter(sub_labels)  # Dict subclass for counting hashable items. Elements are stored as dictionary keys and their counts are stored as dictionary values.
            data_distribution_desc = ""
            for idx, label_name in enumerate(label_order):
                data_distribution_desc = data_distribution_desc  + ", " + str(label_name) + " (" + str(counter[idx]) + ")"
            self.logger.logging("%s labels: %d, labels distributon: %s" %(key, len(sub_labels), data_distribution_desc))

            # data[key] = (sub_main_data, sub_snippets_data)
            data[key] = pd.concat([sub_main_data, sub_snippets_data], axis=1)
            data[key].columns = columns

        # ------------------transform data type--------------------
        extra_params = {}
        if self.embedding == 'glove':
            glove_embedding_matrix, word2idx, idx2word = self.get_embedding_matrix([data["train"], data["val"], data['test']], self.dataset)
            extra_params['embedding_matrix'] = glove_embedding_matrix
            extra_params['word2idx'] = word2idx

        train_data = self.transform_dataframe_to_dict(data["train"], extra_params)
        val_data = self.transform_dataframe_to_dict(data['val'], extra_params)
        test_data = self.transform_dataframe_to_dict(data['test'], extra_params)
        hard_data = self.transform_dataframe_to_dict(data['hard'], extra_params)

        labels = main_data.loc[splits[0],2]        # dtype=int64
        
        if not other_dataset:                   # == other_dataset is false
            # compute_class_weight: estimate class weights for unbalanced dataset, fewer samples with bigger weight
            label_weights = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(label_order)), y=labels).astype(np.float32))
        else:
            label_weights = None
        return train_data, val_data, test_data, hard_data, label_weights, extra_params
        

    def transform_dataframe_to_dict(self, dataframe: pd.DataFrame, extra_params: dict):
        """
        Parameters
        dataframe: `pd.DataFrame` claim_id, claim, label, snippets*10
        extra_params: dict{'embedding_matrix', 'word2idx'}

        Returns
        data: `Dict`
        """
        tokenizer = self.tokenizer
        # iterate by pd.DataFrame.iterrows()/itertuples(), later is more efficient
        claim_ids, claims, labels = [], [], []
        claim_input_ids, claim_masks = [], []
        snippets_input_ids, snippets_token_type_ids, snippets_masks = [], [], []
        claim_adjs, snippets_adjs = [], []
        evd_lengths = []
        claim_lengths = []

        for index, row in dataframe.iterrows():
            claim_id = str(row[0])
            claim = str(row[1])
            label = row[2]
            snippets = row[3:].values.tolist()

            tmp = [len(snippet.split()) for snippet in snippets]
            avg_length = np.mean(tmp)
            if avg_length < 2:
                continue                    # skip claim with no evidence  
            
            claim_ids.append(claim_id)
            claims.append(claim)
            labels.append(label) 
            evd_lengths += tmp
            claim_lengths.append(len(claim.split()))

            if self.model_type == 'get':
                if self.embedding == 'bert':
                    claim_token = tokenizer(claim, return_tensors=None, padding='max_length', truncation=True, max_length=self.claim_length) 
                    snippets_token = tokenizer(snippets, return_tensors=None, padding='max_length', truncation=True, max_length=self.snippet_length) 

                    claim_w2id, claim_mask = claim_token['input_ids'], claim_token['attention_mask']
                    claim_w2id, claim_mask, claim_adj = self.get_tokenizer(claim_w2id, claim_mask, 0, self.claim_length)

                    snippets_w2id, snippets_mask = snippets_token['input_ids'], snippets_token['attention_mask']
                    snippets_w2id, snippets_mask, snippets_adj = self.get_tokenizer(snippets_w2id, snippets_mask, 0, self.snippet_length, )

                elif self.embedding == 'glove':
                    claim_w2id, claim_mask, _ = self.glove_tokenizer(claim, self.claim_length, extra_params['word2idx'])
                    snippets_w2id, snippets_mask,_ = self.glove_tokenizer(snippets, self.snippet_length, extra_params['word2idx'])
                   
                    claim_w2id, claim_mask, claim_adj = self.get_tokenizer(claim_w2id, claim_mask,extra_params['word2idx'][MASK_TOKEN], self.claim_length, window_size=3)
                    snippets_w2id, snippets_mask, snippets_adj = self.get_tokenizer(snippets_w2id, snippets_mask, extra_params['word2idx'][MASK_TOKEN], self.snippet_length)
                
                claim_input_ids.append(claim_w2id)
                claim_masks.append(claim_mask)
                claim_adjs.append(claim_adj)
               
                snippets_input_ids.append(snippets_w2id)   # append item of size (10, length)
                snippets_token_type_ids.append(snippets_mask) # append itemn of size (10, length)
                snippets_masks.append(snippets_mask)
                snippets_adjs.append(snippets_adj)

            elif self.embedding == 'bert':
                claim_token = tokenizer(claim, return_tensors=None, padding='max_length', truncation=True, max_length=self.claim_length)   
                claim_input_ids.append(claim_token['input_ids']) # type: list
                claim_masks.append(claim_token['attention_mask']) # type: list
                
                if self.model_type == 'bert':
                    snippets_token = tokenizer([claim]*10, snippets, return_tensors=None, padding='max_length', truncation=True, max_length=self.snippet_length) 
                else:
                    # 'filer' encded to [101, 5371, 2099, 102]
                    snippets_token = tokenizer(snippets, return_tensors=None, padding='max_length', truncation=True, max_length=self.snippet_length) 
                snippets_input_ids.append(snippets_token['input_ids'])   # append item of size (10, length)
                snippets_token_type_ids.append(snippets_token['token_type_ids']) # append itemn of size (10, length)
                snippets_masks.append(snippets_token['attention_mask'])
            
            elif self.embedding == 'glove':
                claim_w2id, mask, _ = self.glove_tokenizer(claim, self.claim_length, extra_params['word2idx'])
                claim_input_ids.append(claim_w2id)
                claim_masks.append(mask)

                snippets_w2id, mask,_ = self.glove_tokenizer(snippets, self.snippet_length, extra_params['word2idx'])
                snippets_input_ids.append(snippets_w2id)   # append item of size (10, length)
                snippets_token_type_ids.append(mask) # append itemn of size (10, length)
                snippets_masks.append(mask)
        
        if self.model_type == 'get':
            data = {'claim_id': np.array(claim_ids),
                    'claim': np.array(claims),
                    'label': np.array(labels),
                    'claim_input_id': np.array(claim_input_ids), # (n, claim_length)
                    'claim_mask': np.array(claim_masks), # (n, claim_length)
                    'claim_adjs': np.array(claim_adjs, np.float32),

                    'snippets': dataframe.iloc[:, 3:].values,   
                    'snippets_input_id': np.array(snippets_input_ids), # (n, 10, snippet_length)
                    'snippets_token_type_id': np.array(snippets_token_type_ids), # (n, 10, snippet_length)
                    'snippets_mask': np.array(snippets_masks), # (n, 10, snippet_length)
                    'snippets_adjs': np.array(snippets_adjs, np.float32)
                }
        else:
            data = {'claim_id': np.array(claim_ids),
                    'claim': np.array(claims),
                    'label': np.array(labels),
                    'claim_input_id': np.array(claim_input_ids), # (n, claim_length)
                    'claim_mask': np.array(claim_masks), # (n, claim_length)

                    'snippets': dataframe.iloc[:, 3:].values,   
                    'snippets_input_id': np.array(snippets_input_ids), # (n, 10, snippet_length)
                    'snippets_token_type_id': np.array(snippets_token_type_ids), # (n, 10, snippet_length)
                    'snippets_mask': np.array(snippets_masks) # (n, 10, snippet_length)
                    }
        return data

    def filter_websites(self, snippets_data):
        bad_websites = ["factcheck.org", "politifact.com", "snopes.com", "fullfact.org", "factscan.ca"]
        ids = snippets_data.values[:, 0]                # 
        remove_count = 0
        for i, id in enumerate(ids):
            with open("../../multi_fc_publicdata/snippets/" + id, "r", encoding="utf-8") as f:
                lines = f.readlines()

            links = [line.strip().split("\t")[-1] for line in lines]
            remove = [False for _ in range(10)]
            for j in range(len(links)):
                remove[j] = any([bad in links[j] for bad in bad_websites])
            remove = remove[:10]  # 1 data sample has 11 links by mistake in the dataset
            snippets_data.iloc[i, [False] + remove] = "filler"
    
            remove_count += np.sum(remove)
        self.logger.logging("REMOVE COUNT %d" % remove_count)
        return snippets_data

    
    def get_embedding_matrix(self, df_list, dataset, min_occurrence=1, dim=300):
        """
        df_list: list(pd.DataFrame)
        """
        savename = "preprocessed/" + dataset + "_glove.pkl"
        if os.path.exists(savename):
            tmp = pickle.load(open(savename, "rb"))
            glove_embedding_matrix = tmp[0]
            word2idx = tmp[1]
            idx2word = tmp[2]
            return glove_embedding_matrix, word2idx, idx2word

        # glove_vectors = GloVe('840B')
        glove_path = "../../multi_fc_publicdata/glove/glove.6B.%dd.txt"%(dim)

        # dict{str: list}
        # glove_vectors = mz.embedding.load_from_file(file_path=glove_path, mode='glove')._data   # mz.embedding.load_from_file returns Embedding(_data, _output_dim)
        glove_vectors = load_embedding_from_file(file_path=glove_path, mode='glove')

        all_claims = []
        all_snippets = []

        for df in df_list:
            for idx, row in df.iterrows():
                claim = clean_str(str(row[1]))
                label = row[2]
                snippets = row[3:].values.tolist()
                snippets = [clean_str(item) for item in snippets]
                all_claims.append(claim)
                all_snippets += snippets

        all_words = [word for v in all_claims+all_snippets for word in v.split(" ")]
        counter = Counter(all_words)
        all_words = set(all_words)
        all_words = list(set([word for word in all_words if counter[word] > min_occurrence]))
        word2idx = {word: i+2 for i, word in enumerate(all_words)} # reserve 0 for potential mask and 1 for unk token
        word2idx[MASK_TOKEN], word2idx[UNK_TOKEN] = 0, 1
        idx2word = {word2idx[key]: key for key in word2idx}

        num_words = len(idx2word)

        glove_embedding_matrix = np.empty((num_words, 300))
        missed = 0
        for word in word2idx:
            if word in glove_vectors:
                glove_embedding_matrix[word2idx[word]] = glove_vectors[word]
            else:
                # glove_embedding_matrix[word2idx[word]] = np.random.uniform(-0.2, 0.2, size=300)
                glove_embedding_matrix[word2idx[word]] = np.random.normal(size=300)
                missed += 1

        # pickle.dump([glove_embedding_matrix, word2idx, idx2word], open(savename, "wb"))
        print("missed: ", missed)
        return glove_embedding_matrix, word2idx, idx2word

    def glove_tokenizer(self, sentences, max_length:int, word2idx: dict):
        """
        Parameters
        --------------
        sentences: list[str] or str
        
        Returns
        ---------------
        sen_ids: 
        masks: 
        token_type_ids
         
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        sen_ids = []
        masks = []

        for sentence in sentences:
            sen_id = [word2idx[word] if word in word2idx else word2idx[UNK_TOKEN] for word in clean_str(sentence).split()]
            mask = [1 for _ in range(len(sen_id))]
            
            mask.extend([0 for _ in range(max_length - len(sen_id))])
            sen_id.extend([word2idx[MASK_TOKEN] for _ in range(max_length - len(sen_id))])
            
            sen_ids.append(sen_id[:max_length])
            masks.append(mask[:max_length])
        
        if len(sentences) == 1:
            return sen_ids[0], masks[0], masks[0]
        return sen_ids, masks, masks
    
    def get_tokenizer(self, origin_ids, origin_masks, MASK_id: int, max_length:int, window_size=5):
        if not isinstance(origin_ids[0], list):
            origin_ids = [origin_ids]
            origin_masks = [origin_masks]

        sen_ids = []
        masks = []
        adjs = []

        for origin_id, origin_mask in zip(origin_ids, origin_masks):
            length = sum(origin_mask)
            sen_id = origin_id[:length]

            words_list = list(set(sen_id))       # remove duplicate words in original order
            words_list.sort(key=sen_id.index)
            words2id = {word: id for id, word in enumerate(words_list)}

            length_ = len(words2id)
            neighbours = [set() for _ in range(length_)]
        
            for i, word in enumerate(sen_id):
                for j in range(max(i-window_size+1, 0), min(i+window_size, length)):
                    neighbours[words2id[word]].add(words2id[sen_id[j]])

            # gnn graph
            words_list.extend([MASK_id for _ in range(max_length-length_)])
            adj = [[1 if (max(i, j) < length_) and (j in neighbours[i]) else 0 for j in range(max_length)] for i in range(max_length)]
            mask = [1 if i<length_ else 0 for i in range(max_length)]

            sen_ids.append(words_list[:max_length])
            masks.append(mask[:max_length])
            adjs.append(_laplacian_normalize(np.array(adj)))
        
        if len(origin_ids) == 1:
            return sen_ids[0], masks[0], adjs[0]
        return sen_ids, masks, adjs

def _laplacian_normalize(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)).A


def load_embedding_from_file(file_path: str, mode: str = 'word2vec'):
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec', 'fasttext'
        or 'glove'.(default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    embedding_data = {}
    output_dim = 0
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r') as f:
            output_dim = int(f.readline().strip().split(' ')[-1])
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
    elif mode == 'glove':
        with open(file_path, 'r') as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = np.array([float(val) for val in current_line[1:]])
                
    else:
        raise TypeError(f"{mode} is not a supported embedding type."
                        f"`word2vec`, `fasttext` or `glove` expected.")
    return embedding_data