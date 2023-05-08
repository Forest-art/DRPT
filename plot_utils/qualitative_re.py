import torch
import numpy as np
import cv2
import os
import torch.nn as nn



os.makedirs("demos", exist_ok=True)
def quali_result(logits, data, pairs_dataset):
    batch_attr, batch_obj, batch_target, image = data[1], data[2], data[3], data[4]
    # image_path = "/nvme/hantao/lxc/dataset/cgqa/images/" + image[0]
    # img = cv2.imread(image_path)
    logits = logits.type(torch.float)
    # softmax = nn.Softmax(dim=1)
    # logits = softmax(logits)
    # values, indices = logits.topk(k=5, dim=1)
    # print(values, indices, batch_target)
    # if indices[0][0]==batch_target[0]:
    #     print(pairs_dataset[indices[0][0]], pairs_dataset[indices[0][1]], pairs_dataset[indices[0][2]], pairs_dataset[indices[0][3]], pairs_dataset[indices[0][4]], pairs_dataset[batch_target[0]], image[0])
    #     cv2.imwrite("./demos/" + image_path.split('/')[-2] + "_" + image_path.split('/')[-1], img)


    see = [np.random.randint(len(pairs_dataset)) for i in range(10)]
    for j in see:
    # j = 30
        values, indices = logits[:, j].topk(k=5)
        print(pairs_dataset[j], pairs_dataset[batch_target[indices[0]]])
        print(image[indices[0]], image[indices[1]], image[indices[2]], image[indices[3]], image[indices[4]])
        # print(logits.shape)
        for i in range(5):
            image_path = "/nvme/hantao/lxc/dataset/ut-zappos/images/" + image[indices[i]]
            # image_path = "/nvme/hantao/lxc/dataset/cgqa/images/" + image[0]
            img = cv2.imread(image_path)
            os.makedirs("demos/" + pairs_dataset[j][0] + '_' + pairs_dataset[j][1], exist_ok=True)
            cv2.imwrite("./demos/" + pairs_dataset[j][0] + '_' + pairs_dataset[j][1] + "/" + image_path.split('/')[-2] + "_" + image_path.split('/')[-1], img)


