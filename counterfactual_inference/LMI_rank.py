import math
import pickle
import argparse

from collections import Counter
import statistics
import pandas as pd


def merge_label(label: str, *args):
    dataset = args[0]
    label_order = args[1]
    index = label_order.index(label)
    if dataset == 'snes':
        return 'false' if index < 2 else 'true'         # merge false, mostly false as false
    else:
        return 'false' if index < 3 else 'true'         # merge pants on fire, false, mostly false as false

if __name__ == '__main__':
    parser = argparse.ArgumentParser("description")

    parser.add_argument("--dataset", type=str, default="snes", help="Dataset")
    parser.add_argument("--merge", type=int, default=0, help="Merge labels if merge is set to 1")
    parser.add_argument("--n", type=int, default=1,help="n of N-gram")
    parser.add_argument("--top_k", type=int, default=10, help="Top k")

    args = parser.parse_args()

    

    new_label_order = ['false', 'true']
    path_prefix = "../../multi_fc_publicdata/" + args.dataset + "/" + args.dataset
    main_data = pd.read_csv("%s.tsv" % (path_prefix), sep='\t', header=None)
    snippets_data = pd.read_csv("%s_snippets.tsv" % (path_prefix), sep='\t', header=None )
    label_order = pickle.load(open("%s_labels.pkl" %(path_prefix), "rb")) # 1-d list
    splits = pickle.load(open("%s_index_split.pkl" % (path_prefix), "rb")) # 2-d list

    if args.merge == 1:
        main_data.iloc[:,2] = main_data.iloc[:, 2].apply(merge_label, args=(args.dataset, label_order))
        label_order = new_label_order
    main_data = main_data.iloc[splits[0]]

    total_dic = {label: {} for label in label_order}
    total_n_gram = {}
    all_labels = main_data.values[:, 2]
    total_label = Counter(all_labels)  # Dict subclass for counting hashable items. Elements are stored as dictionary keys and their counts are stored as dictionary values.
    

    for index, line in main_data.iterrows():
        claim = str(line[1]).lower().split()    # list
        label = str(line[2])
        # print(claim, label)
        dic = total_dic[label]
        for i in range(len(claim)-args.n+1):
            n_gram = " ".join(claim[i:i+args.n])
            # print(n_gram, label)
            dic[n_gram] = dic.get(n_gram, 0) + 1
            total_n_gram[n_gram] = total_n_gram.get(n_gram, 0) + 1

    D = sum(total_n_gram.values())

    for label in label_order:
        dic = total_dic[label]
        
        # n_gram: tuple(LMI, p(l|w))
        statistic = {n_gram: (c1/D*math.log(c1*D/total_label[label]/total_n_gram[n_gram]), c1/total_n_gram[n_gram]) for n_gram, c1 in dic.items()}
        statistic = sorted(statistic.items(), key= lambda item: item[1][1], reverse=True) # sort by LMI
        print("-"*10 + label +"-"*10 + "number of n_gram: %d" %(len(dic)))
        for item in statistic[:args.top_k]:
            print(item)
        print()