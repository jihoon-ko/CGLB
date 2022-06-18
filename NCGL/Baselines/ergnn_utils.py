import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import copy


class MF_sampler(nn.Module):
    # sampler for ERGNN MF and MF*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps)
        else:
            return self.sampling(ids_per_cls_train, budget, feats)

    def sampling(self,ids_per_cls_train, budget, vecs):
        centers = [vecs[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1,-1).mm(vecs[ids_per_cls_train[i]].permute(1,0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            nearest = rank[i][0:min(budget, len(ids_per_cls_train[i]))]
            ids_selected.extend([ids[i] for i in nearest])
        return ids_selected

class CM_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        #n#eeds a line ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        budget_dist_compute = 1000
        vecs = vecs.half()
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = ids_per_cls_train[i] if len(ids_per_cls_train[i])<budget_dist_compute else random.choices(ids_per_cls_train[i], k=budget_dist_compute)
            dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist,dim=-1) # include distance to all the other classes
            n_selected = (dist_<d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = rank[:budget]
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected

class random_sampler(nn.Module):
    # sampler for ERGNN CM and CM*
    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def forward(self, ids_per_cls_train, budget, feats, reps, d):
        if self.plus:
            return self.sampling(ids_per_cls_train, budget, reps, d)
        else:
            return self.sampling(ids_per_cls_train, budget, feats, d)

    def sampling(self,ids_per_cls_train, budget, vecs, d):
        ids_selected = []
        for i,ids in enumerate(ids_per_cls_train):
            ids_selected.extend(random.sample(ids,min(budget,len(ids))))
        return ids_selected