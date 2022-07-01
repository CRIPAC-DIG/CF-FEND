import os
import sys
import math
import time
import json
import torch
import random

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from Models import *
from sklearn import metrics
from datetime import datetime
from utility.load_data import *
from utility.logger import Logger
from utility.parser import parse_args
from torch.utils.data import DataLoader, Dataset

eva_metrics = ['f1_macro',
            'f1_micro',
            ]

def transformer_collate(batch):
    # batch sample consists of [claimID, claim, label, snippets]

    claimIDs = [t[0] for t in batch]
    claims = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    snippets = [t[3] for t in batch]

    return  claimIDs, claims, labels, snippets

class dataset(Dataset):
    
    def __init__(self, data: dict) -> None:
        super().__init__()
        
        self.claim_id = data['claim_id']                     # numpy.ndarray
        self.claim = data['claim']         
        self.label = torch.from_numpy(data['label'])         # torch.tensor
        self.claim_input_id = torch.from_numpy(data['claim_input_id'])
        self.claim_mask = torch.from_numpy(data['claim_mask'])

        self.snippets = data['snippets']                  # numpy.ndarray
        self.snippets_input_id = torch.from_numpy(data['snippets_input_id']) # 'torch.tensor': (n, 10, snippet_length)
        self.snippets_token_type_id = torch.from_numpy(data['snippets_token_type_id'])
        self.snippets_mask = torch.from_numpy(data['snippets_mask'])
        self.length = len(self.claim_id)

    def __len__(self,):
        return self.length

    def __getitem__(self, index) :
        return self.claim_input_id[index], self.claim_mask[index],\
                self.snippets_input_id[index] , self.snippets_token_type_id[index], self.snippets_mask[index], \
                self.label[index]
                # self.claim_id[index], self.claim[index], self.snippets[index]

def custom_collect_fn(batch):
    claims = [t[0] for t in batch]         # total batch -> mini_batch
    snippets = [t[1] for t in batch]
    labels = np.array([t[2] for t in batch])

    return  claims, snippets, torch.from_numpy(labels)

    
class Trainer():

    def __init__(self, args, logger: Logger) -> None:
        self.logger = logger
        self.logger.logging("PID: %d" % os.getpid())
        args_setting = json.dumps(vars(args), sort_keys=True, indent=2)
        self.logger.logging("============= Debiasing Fake News via Counterfactual Inference========================\n{}".format(args_setting))
        
        # ----- init parameters --------
        self.args = args
        self.lr = args.lr
        self.dataset = args.dataset
        self.model_type = args.model
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.verbose = args.verbose
        self.early_stopping_patience = args.early_stopping_patience
        self.claim_length = args.claim_length
        self.snippet_length = args.snippet_length
        self.label_num = args.label_num
        self.embedding = args.embedding
        self.up_bound = args.up_bound

        self.cuda = torch.cuda.is_available()

        # ----- load data -------------
        self.logger.logging('Loading Data')
        My_Load_Data = Load_Data(args, self.logger)
        train_data, val_data, test_data, hard_data, label_weights, self.extra_params = My_Load_Data.load_data()
        self.label_weights = label_weights

        self.logger.logging('Loading Done.')
        self.train_dataset = dataset(train_data)
        self.val_dataset = dataset(val_data) 
        self.test_dataset = dataset(test_data)
        self.hard_dataset = dataset(hard_data)

        self.logger.logging("Train Dataset: %d" %(len(self.train_dataset)))
    
    def train(self, fold):
        self.logger.logging("============= Fold %d ========================" %(fold))
        # self.saved_model = 'result/' + '.'.join([str(v) for v in [self.model_type, self.dataset, self.label_num, 'fold_%d'%fold]]) + '.pkl'
        self.saved_model = 'result/' + '.'.join([str(v) for v in [self.model_type, self.dataset, self.label_num]]) + '.pkl'
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.batch_size,
                                    #    collate_fn=custom_collect_fn,    # default_fn: input must be tensor
                                       shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False)
        self.hard_loader = DataLoader(self.hard_dataset, 
                                      batch_size=self.batch_size,
                                      shuffle=False)
        
        if self.model_type == 'bert':
            self.model = Bert_Model(self.args)
        elif self.model_type == 'mac':
            self.model = MAC(self.args, self.extra_params)

        if self.cuda:
            self.logger.logging("CUDA")
            self.model.cuda()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr= args.lr)
        self.lr_scheduler = self.set_lr_scheduler()

        model = self.model
        criterion = nn.CrossEntropyLoss(weight=self.gpu(self.label_weights))
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        best_f1_macro = 0.0
        best_acc = 0.0
        patience_step = 0

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.logging("model parameters: %d" % params)

        for epoch in range(self.num_epochs):

            epoch_loss = 0.0
            acc_vector = []
            t1 = time.time()
            model.train()

            for batch, (claim, claim_mask, snippets, snippets_token_type, snippets_mask, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                optimizer.zero_grad()

                claim, claim_mask, snippets, snippets_token_type, snippets_mask, labels = self.gpu(claim),self.gpu(claim_mask), \
                        self.gpu(snippets), self.gpu(snippets_token_type), self.gpu(snippets_mask), self.gpu(labels)
                class_logit = model(claim, claim_mask, snippets, snippets_token_type, snippets_mask)
                class_loss = criterion(class_logit, labels)

                epoch_loss += class_loss.item() 
                argmax = torch.max(class_logit, dim=1)[1]
                acc_vector.append((labels == argmax.squeeze()).float().mean().item())
                loss = class_loss
                if math.isnan(loss) == True:
                    self.logger.logging('ERROR: loss is nan')
                    sys.exit()

                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            t2 = time.time()
            epoch_train_time = t2 - t1
            with torch.no_grad():
                res_val = self.evaluate(dataloader=self.val_loader)

            f1_macro_val,f1_micro_val = res_val['f1_macro'], res_val['f1_micro']
            self.logger.logging("Epoch %03d | Train time: %04.1f(s) | Train loss: %.4f | Train_acc: %.4f "
                                "| Val F1_macro = %.5f | Val F1_micro = %.5f | Val Accuracy = %.4f "%(epoch,
                                epoch_train_time, epoch_loss, np.mean(acc_vector), f1_macro_val,
                                f1_micro_val, f1_micro_val))

            if self.verbose and (epoch + 1) % self.verbose == 0:
                with torch.no_grad():
                    res_test = self.evaluate(dataloader=self.test_loader)
                f1_macro_test, f1_micro_test  = res_test['f1_macro'],res_test['f1_micro']
                self.logger.logging("Epoch %03d ｜Test F1_macro = %.5f | Test F1_micro = %.5f" %(epoch,
                                f1_macro_test, f1_micro_test))
            
            if f1_macro_val > best_f1_macro:
                best_f1_macro = f1_macro_val
                patience_step = 0

                torch.save(model.state_dict(), self.saved_model)
            else:
                patience_step += 1
            if patience_step > self.early_stopping_patience:
                self.logger.logging("Early Stop due to no better f1_macro!")
                break
        
        return self.load_best_model(), self.load_best_model_debias()    # tuple(sub_tuple(), sub_tuple())
    
    
    def load_best_model(self):
        model = self.model
        model.load_state_dict(torch.load(self.saved_model))
        model.eval()
        with torch.no_grad():
            res_val = self.evaluate(dataloader=self.val_loader)
            res_test = self.evaluate(dataloader=self.test_loader)
            res_hard = self.evaluate(dataloader=self.hard_loader)
            # res_other_hard = self.evaluate(dataloader=self.other_hard_loader)
        f1_macro_val = res_val['f1_macro']
        f1_macro_test = res_test['f1_macro']
        f1_macro_hard = res_hard['f1_macro']
        # f1_macro_other_hard = res_other_hard['f1_macro']
        
        self.logger.logging("Best Val F1_macro = %.4f ｜Test F1_macro = %.5f | Hard F1_macro = %.5f"%(
                            f1_macro_val, f1_macro_test, f1_macro_hard))
        return res_test
    
    def load_best_model_debias(self):
        
        model = self.model
        model.load_state_dict(torch.load(self.saved_model))
        model.eval()

        with torch.no_grad():
            avg = self.set_avg(self.train_loader)
            model.set_avg(avg)
            
        best_lamda = 0.0
        best_res_test, best_res_hard, best_res_other_hard = None, None, None
        res = {}
        best_val_f1_macro = 0.0 
        count_patience_epochs = 0
        model.train(False)
        for lamda in range(0, self.up_bound*100+1, 10):
            lamda = lamda /100
            with torch.no_grad():
                res_val = self.evaluate(dataloader=self.val_loader, debias=lamda)
                res_test = self.evaluate(dataloader=self.test_loader, debias=lamda)
                res_hard = self.evaluate(dataloader=self.hard_loader, debias=lamda)
            self.logger.logging('| Lamda = %.3f '
                        '| Val F1_macro = %.5f '
                        '| Test F1_macro = %.5f | Test F1_micro = %.5f '
                        '| Hard F1_macro = %.5f | Hard F1_micro = %.5f '
                        % (lamda, res_val['f1_macro'], res_test['f1_macro'], res_test['f1_micro'], 
                        res_hard['f1_macro'], res_hard['f1_micro']))
        
            res[lamda] = (res_test, res_hard)       # dict{float: tuple(dict, dict, dict)}
            if res_val['f1_macro'] > best_val_f1_macro:
                best_val_f1_macro = res_val['f1_macro']
                count_patience_epochs = 0
            else:
                count_patience_epochs += 1
    
        return res

    def evaluate(self, dataloader: DataLoader, debias:int=0):
        """
        
        returns
        ----------
        res: dict (keys include "f1_macro", "f1_micro", "accuracy", "f1", ...)
        """
        model = self.model
        model.eval()

        predict_label = []
        gold_label = []
        for batch, (claim, claim_mask, snippets, snippets_token_type, snippets_mask, labels) in enumerate(dataloader):
            claim, claim_mask, snippets, snippets_token_type, snippets_mask = self.gpu(claim),self.gpu(claim_mask), \
                        self.gpu(snippets), self.gpu(snippets_token_type), self.gpu(snippets_mask)
            class_prob = model(claim, claim_mask, snippets, snippets_token_type, snippets_mask, debias)
            
            argmax = torch.max(class_prob, dim=1)[1]   
            gold_label.extend(labels.cpu().detach().numpy())         # list type
            predict_label.extend(argmax.cpu().detach().numpy())

        res = self.computing_metrics(gold_label, predict_label)
        return res

    def computing_metrics(self, true_labels: list, predict_labels: list):
        """
        Computing classification metrics for 2 category classification

        Parameters
        ----------
        true_labels: :class:`list`
        predict_labels: :class:`list`

        """
    
        assert len(true_labels) == len(predict_labels)

        accuracy = metrics.accuracy_score(true_labels, predict_labels)
        f1_macro = metrics.f1_score(true_labels, predict_labels, average="macro")
        f1_micro = metrics.f1_score(true_labels, predict_labels, average="micro")
        # f1 = metrics.f1_score(true_labels, predict_labels)

        # precision_true = metrics.precision_score(true_labels, predict_labels, labels=[0], average=None)[0]
        # recall_true = metrics.recall_score(true_labels, predict_labels, labels=[0], average=None)[0]
        # f1_macro_true = metrics.f1_score(true_labels, predict_labels, labels=[0], average=None)[0]
        # precision_false = metrics.precision_score(true_labels, predict_labels, labels=[1], average=None)[0]
        # recall_false = metrics.recall_score(true_labels, predict_labels, labels=[1], average=None)[0]
        # f1_macro_false = metrics.f1_score(true_labels, predict_labels, labels=[1], average=None)[0]

        result = {
            # total
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            # 'f1': f1,
            # 'accuracy': accuracy,

            # true class
            # "precision_true": precision_true,
            # "recall_true": recall_true,
            # "f1_macro_true": f1_macro_true,

            # false class
            # "precision_false": precision_false,
            # "recall_false": recall_false,
            # "f1_macro_false": f1_macro_false
        }
        return result

    def set_avg(self, loader):

        model = self.model
        model.eval()
        total_num = 0
        feature_sum = self.gpu(torch.tensor([0]))
        for batch, (claim, claim_mask, snippets, snippets_token_type, snippets_mask, labels) in enumerate(loader):
            claim, claim_mask, snippets, snippets_token_type, snippets_mask, labels = self.gpu(claim),self.gpu(claim_mask), \
                        self.gpu(snippets), self.gpu(snippets_token_type), self.gpu(snippets_mask), self.gpu(labels)
            feature = model(claim, claim_mask, snippets, snippets_token_type, snippets_mask, evd_output=True)
            feature_sum = feature_sum + torch.sum(feature, dim=0)
            total_num += claim.shape[0]

        feature_avg =feature_sum / total_num
        return feature_avg

    def set_lr_scheduler(self):
        lr_lambda = lambda epoch:  1/(1. + 10 * (epoch/100)) ** 0.75            # lr' = lr * lr_lambda(epoch)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        return scheduler
    
    def gpu(self, x: torch.tensor):
        return  x.cuda() if self.cuda else x


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu

def compute_average_results(kfold_results):
    """
    Compute average results of every metric in list of metrics
    """
    k = len(kfold_results)
    avg_results = {}
    for metric in eva_metrics:
        ls = [fold[metric] for fold in kfold_results]
        avg, std = np.mean(ls), np.std(ls)
        avg_results[metric] = {"avg": avg, "std": std, "all_%d_folds"%(k): " ".join([str(e) for e in ls])}
    return avg_results

def compute_average_debias_results(kfold_debias_results):
    """
    Compute average results of every metric in list of metrics

    Parameters:
    kfold_debias_results： list[ dict{float: tuple(dict, dict, dict)} ]
    """
    k = len(kfold_debias_results)
    kfold_test, kfold_hard = {}, {}
    avg_results = {}
    results = ''
    
    for res_debias in kfold_debias_results:
        for lamda, (res_test, res_hard) in res_debias.items():
            kfold_test[lamda] = kfold_test.get(lamda, [])
            kfold_hard[lamda] = kfold_hard.get(lamda, [])
            # kfold_other_hard[lamda] = kfold_other_hard.get(lamda, [])

            kfold_test[lamda].append(res_test)
            kfold_hard[lamda].append(res_hard)
            # kfold_other_hard[lamda].append(res_other_hard)

    for lamda in kfold_test.keys():
        avg_test = compute_average_results(kfold_test[lamda])
        avg_hard = compute_average_results(kfold_hard[lamda])
        # avg_other_hard = compute_average_results(kfold_other_hard[lamda])
        results += "Lamda = %.2f | " % (lamda)
        for key, avg in zip(['Test', 'Hard'], [avg_test, avg_hard]):
            results += key + " "
            for metric in eva_metrics:
                results += "%s = %.4f %.4f |" % (metric, avg[metric]['avg'], avg[metric]['std'])
        results += '\n'

    return results

if __name__ == '__main__':
    kfold_test_results = []
    kfold_debias_results = []
    parser = parse_args()
    args = parser.parse_args()
    task_name = "%s" %(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger = Logger(filename=task_name, is_debug=args.debug)
    set_seed(args.seed)

    trainer = Trainer(args, logger, )
    for fold in range(args.num_fold):
        res_test, res_debias = trainer.train(fold)
        kfold_test_results.append(res_test)
        kfold_debias_results.append(res_debias)

        args.seed = args.seed + 100
        set_seed(args.seed)

    logger.logging("Average debias results from %d folds"%(args.num_fold))
    avg_debias_results = compute_average_debias_results(kfold_debias_results)
    logger.logging('\n' + avg_debias_results)
    
    
