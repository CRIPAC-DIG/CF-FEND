import argparse
from email.policy import default

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # snes: snopes; pomt: politifact
    
    # train realated
    parser.add_argument('--debug', action='store_true', help='') # including this argument in cmd means True
    parser.add_argument('--num_fold', type=int, default=1, help='')
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--verbose", default=5, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--up_bound", type=int, default=2)

    parser.add_argument('--fold', type=int, default=0, help='')
    
    # dataset related
    parser.add_argument("--dataset", default="pomt", choices=["snes", "pomt"], type=str)
    parser.add_argument("--filter_websites", default=0, type=int)
    parser.add_argument("--label_num", default=6, type=int)
    parser.add_argument("--claim_length", default=100, type=int)
    parser.add_argument("--snippet_length", default=100, type=int)
    parser.add_argument("--filter_mixture", default=0, type=int, help='')


    # model
    parser.add_argument("--model", default="bert", choices=["bert", "declare", "mac", "get"], type=str)
    parser.add_argument("--embedding", default="bert", choices=['bert', 'glove'], type=str, help='')
    parser.add_argument('--config', type=str, default='bert-base-uncased')
    parser.add_argument('--bert_cache', type=str, default='../transformers/')
    parser.add_argument("--dropout", type=float, default=0.2)


    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--lstm_layers", default=0, type=int)
    parser.add_argument("--num_att_heads_for_words", default=1, type=int)
    parser.add_argument("--num_att_heads_for_evds", default=1, type=int)
    parser.add_argument("--use_claim_source", default=0, type=int)
    parser.add_argument("--use_evd_source", default=0, type=int)

    return parser
