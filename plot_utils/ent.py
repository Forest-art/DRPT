import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



def analy_ent(all_logits, all_pair_gt, pairs_dataset):
    ent_attr, ent_obj = pairs_dataset.ent_attr, pairs_dataset.ent_obj
    # print(all_logits.shape, all_pair_gt.shape)
    values, indices = all_logits.topk(k=1, dim=1)
    indices = indices.squeeze()

    # acc = torch.sum(indices == all_pair_gt) / len(all_pair_gt)
    # print(acc, len(pairs_dataset.pairs))

    acc_list = []
    ent_list = []

    for i in range(len(pairs_dataset.pairs)):
        pos_sum = 0
        i_sum = 0
        for j in range(len(all_pair_gt)):
            if all_pair_gt[j] == indices[j] and all_pair_gt[j] == i:
                pos_sum += 1
            if all_pair_gt[j] == i:
                i_sum += 1
        if i_sum != 0:
            i_acc = pos_sum / i_sum
        else:
            i_acc = 0
        acc_list.append(i_acc)
        (att, obj) = pairs_dataset.pairs[i]
        ent_ao = ent_attr[att] * ent_obj[obj]
        ent_list.append(ent_ao)
        # print(acc_list, ent_list)
    
    acc = np.array(acc_list)
    ent = np.array(ent_list)
    plt.plot(ent,acc, 'bo-', label="DRPT (6.2)")
    plt.savefig('demos/Acc-Ent.pdf', dpi=300)







        



    


