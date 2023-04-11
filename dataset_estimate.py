import torch
import os
import numpy as np
import csv
import pickle
import shutil

data_path = "/data/jyy/lll/dataset/mit-states/"



def parse_split():
    def parse_pairs(pair_list):
        with open(pair_list, 'r') as f:
            pairs = f.read().strip().split('\n')
            # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
            pairs = [t.split() for t in pairs]
            pairs = list(map(tuple, pairs))
        attrs, objs = zip(*pairs)
        return attrs, objs, pairs

    tr_attrs, tr_objs, tr_pairs = parse_pairs(
        '%s/%s/train_pairs.txt' % (data_path, 'compositional-split-natural'))
    vl_attrs, vl_objs, vl_pairs = parse_pairs(
        '%s/%s/val_pairs.txt' % (data_path, 'compositional-split-natural'))
    ts_attrs, ts_objs, ts_pairs = parse_pairs(
        '%s/%s/test_pairs.txt' % (data_path, 'compositional-split-natural'))

    all_attrs, all_objs = sorted(
        list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
    all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
    return all_attrs, all_objs, all_pairs


if __name__=="__main__":
    data = torch.load(data_path + "/metadata_compositional-split-natural.t7")
    all_attrs, all_objs, all_pairs = parse_split()
    ent_attr, ent_obj = {}, {}
    for attr in all_attrs:
        ent_attr[attr] = 0
    for obj in all_objs:
        ent_obj[obj] = 0
    for (attr, obj) in all_pairs:
        ent_attr[attr] += 1
        ent_obj[obj] += 1
    print(ent_attr, ent_obj)
        