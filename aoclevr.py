import torch
import os
import numpy as np
import csv
import pickle
import shutil

data_path = "/data/jyy/lll/dataset/vaw-czsl/"
# csv_path = data_path + "objects_metadata.csv"
# csv_reader = csv.reader(open(csv_path))
# pkl = pickle.load(open(data_path + "/metadata_pickles/metadata_ao_clevr__UV_random__comp_seed_2000__seen_seed_0__test.pkl", 'rb'), encoding='utf-8')

def process_data():

    ### vaw-czsl
    with open(os.path.join(data_path, "compositional-split-natural", "test_pairs.txt"), 'r') as f:
        pairs = f.readlines()
        # print(pairs)
        with open(os.path.join(data_path, "compositional-split-natural", "test_pairs.txt"), 'w') as g:
            for pair in pairs:
                g.writelines(pair.replace('+', ' '))













    ### aoclevr
    # img_list = os.listdir(data_path + "images")
    # print(len(csv_reader))
    # for idx, line in enumerate(csv_reader):
    #     if idx < 1:
    #         continue
    #     attr = line[3]
    #     obj = line[2]
    #     img_path = os.path.join(data_path, "images", line[0])
    #     print(img_path, attr, obj)
    #     os.makedirs(os.path.join(data_path,"images",attr+"_"+obj), exist_ok=True)
    #     shutil.move(img_path, os.path.join(data_path,"images",attr+"_"+obj, line[0]))

    # for key in pkl:
    #     print(key)
    # print(pkl["seen_pairs"], pkl["unseen_closed_test_pairs"], pkl["unseen_closed_val_pairs"])
    # with open("/data/jyy/lll/dataset/aoclevr/compositional-split-natural/"+"val_pairs.txt", "w") as f:
    #     for a,o in pkl["unseen_closed_val_pairs"]:
    #         f.write(a)
    #         f.write(" ")
    #         f.write(o)
    #         f.write("\n")

    # all_data = []
    # for data in pkl["train_data"]:
    #     instance = {}
    #     instance["image"] = data[1] + "_" + data[2] + "/" + data[0]
    #     instance["attr"] = data[1]
    #     instance["obj"] = data[2]
    #     instance["set"] = "train"
    #     all_data.append(instance)
    # for data in pkl["val_data"]:
    #     instance = {}
    #     instance["image"] = data[1] + "_" + data[2] + "/" + data[0]
    #     instance["attr"] = data[1]
    #     instance["obj"] = data[2]
    #     instance["set"] = "val"
    #     all_data.append(instance)
    # for data in pkl["test_data"]:
    #     instance = {}
    #     instance["image"] = data[1] + "_" + data[2] + "/" + data[0]
    #     instance["attr"] = data[1]
    #     instance["obj"] = data[2]
    #     instance["set"] = "test"
    #     all_data.append(instance)
    #     # print(instance)
    # torch.save(all_data, "/data/jyy/lll/dataset/aoclevr/metadata_compositional-split-natural.t7")


    # data = torch.load("/data/jyy/lll/dataset/mit-states/metadata_compositional-split-natural.t7")
    # print(data)
    # for instance in data:
    #     print(instance)


if __name__=="__main__":
    process_data()