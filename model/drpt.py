from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from model.common import *
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
from visualazition import *


class DRPT(nn.Module):
    def __init__(self, config, attributes, classes, offset, ent_attr, ent_obj):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.dtype = torch.float16
        self.dropout = nn.Dropout(config.dropout)
        self.ent_attr = torch.Tensor(list(ent_attr.values()))
        self.ent_obj = torch.Tensor(list(ent_obj.values()))
        self.avg_ent_att, self.avg_ent_obj = self.ent_attr.mean(), self.ent_obj.mean()
        self.token_ids, self.soft_att, self.soft_obj, self.soft_prompt = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        self.train_status = "state+object"      ###status in ["object", "state", "object+state"]
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype, self.attributes, self.classes)
        for p in self.parameters():
            p.requires_grad=False
        # self.soft_att_dict, self.soft_obj_dict = nn.ParameterDict({}), nn.ParameterDict({})
        # self.decompose_attr_obj()
        self.soft_att_obj = nn.ParameterDict({'att': nn.Parameter(self.soft_att), 'obj': nn.Parameter(self.soft_obj)})
        self.soft_att_fix = self.soft_att_obj['att'].detach().cuda()
        self.soft_obj_fix = self.soft_att_obj['obj'].detach().cuda()
        if self.config.update==True:
            self.update_status(0)


    def decompose_attr_obj(self):
        #### Rename soft_attr and soft_obj
        for id, tok in enumerate(self.attributes):
            self.soft_att_dict[tok] = nn.Parameter(self.soft_att[id])
        for id, tok in enumerate(self.classes):
            self.soft_obj_dict[tok] = nn.Parameter(self.soft_obj[id])

    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x", context_length=self.config.context_length).cuda()

        #### Construct the prompt for attributes
        tokenized_attr = torch.cat([clip.tokenize(tok, context_length=self.config.context_length) for tok in self.attributes])
        orig_token_embedding_attr = self.clip.token_embedding(tokenized_attr.cuda())
        soft_att = torch.zeros(
            (len(self.attributes), orig_token_embedding_attr.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding_attr):
            eos_idx = tokenized_attr[idx].argmax()
            soft_att[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        #### Construct the prompt for objects
        tokenized_obj = torch.cat([clip.tokenize(tok, context_length=self.config.context_length) for tok in self.classes])
        orig_token_embedding_obj = self.clip.token_embedding(tokenized_obj.cuda())
        soft_obj = torch.zeros(
            (len(self.classes), orig_token_embedding_obj.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding_obj):
            eos_idx = tokenized_obj[idx].argmax()
            soft_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        #### Construct the prompt for the prefix.
        prefix_init = "a photo of"
        n_ctx = len(prefix_init.split())
        prompt = clip.tokenize(prefix_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        prefix_vectors = embedding[0, 1 : 1 + n_ctx, :]

        return token_ids, soft_att, soft_obj, prefix_vectors



    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)

        soft_att = self.dropout(self.soft_att_obj['att'])
        soft_obj = self.dropout(self.soft_att_obj['obj'])

        eos_idx = int(self.token_ids[0].argmax())
        
        # soft_att = torch.stack([self.soft_att_dict[key] for index, key in enumerate(self.soft_att_dict)])
        # soft_obj = torch.stack([self.soft_obj_dict[key] for index, key in enumerate(self.soft_obj_dict)])
        # soft_att = self.dropout(soft_att)
        # soft_obj = self.dropout(soft_obj)
        token_tensor[:, eos_idx - 2, :] = soft_att[attr_idx].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_obj[obj_idx].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        # x_mlp = self.mlp(x.type(torch.float)).type(self.clip.dtype)
        # x = self.weight * x + (1 - self.weight) * x_mlp

        normalized_x = x / x.norm(dim=-1, keepdim=True)
        return normalized_x


    def ent_weight(self, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        w_att = self.ent_attr
        w_obj = self.ent_obj
        ent_weight = torch.zeros(len(idx))
        ent_weight = w_att[att_idx] * w_obj[obj_idx]
        ent_weight = ent_weight / ent_weight.max()
        # print(ent_weight)
        return 2 - 1 *  ent_weight


    def update_status(self, epoch):
        if epoch // self.config.epoch_round % 3 == 0:
            self.train_status = "object"
        elif epoch // self.config.epoch_round % 3 == 1:
            self.train_status = "state"
        else:
            if self.train_status != "state+object":
                self.soft_att_fix = self.soft_att_obj['att'].detach().cuda()
                self.soft_obj_fix = self.soft_att_obj['obj'].detach().cuda()
            self.train_status = "state+object"

        if self.train_status == "object":
            self.soft_att_obj['att'].requires_grad = False
            self.soft_att_obj['obj'].requires_grad = True
        elif self.train_status == "state":
            self.soft_att_obj['att'].requires_grad = True
            self.soft_att_obj['obj'].requires_grad = False
        else:
            self.soft_att_obj['att'].requires_grad = True
            self.soft_att_obj['obj'].requires_grad = True



    def forward(self, batch, idx):
        batch_img, batch_attr, batch_obj, batch_target = batch
        batch_img, batch_attr, batch_obj, batch_target = batch_img.cuda(), batch_attr.cuda(), batch_obj.cuda(), batch_target.cuda()
        b = batch_img.shape[0]

        #### Image Encoder
        batch_img = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768

        #### Text Encoder
        token_tensors = self.construct_token_tensors(idx)
        text_features = self.text_encoder(self.token_ids, token_tensors, enable_pos_emb=self.enable_pos_emb, idx=idx)

        ent_weight = self.ent_weight(idx).cuda().type(self.dtype)
        #### Compute Logits and loss
        logits = self.clip.logit_scale.exp() * batch_img @ text_features.t()  

        loss_reg = 0
        if self.train_status == "state+object":
            loss_reg = (self.soft_att_fix - self.soft_att_obj['att']).norm(p=1) + (self.soft_obj_fix - self.soft_att_obj['obj']).norm(p=1)
        else:
            loss_reg = 0

        # loss = self.loss_fn(logits, batch_target)
        if self.config.ent_weight == True:
            loss = F.cross_entropy(logits, batch_target, weight=ent_weight) + 0 * loss_reg
        else:
            loss = F.cross_entropy(logits, batch_target) + 0 * loss_reg


        return logits, loss
