# most of the gem code are modifed from facebook's official repo
import pickle

import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
#import quadprog
from .gem_utils import store_grad, overwrite_grad, project2cone2


class NET(nn.Module):
    """
    wrap the base_model to be a lifelong model
    """    
    def __init__(self,
                model,
                task_manager,
                args):
        super(NET, self).__init__()        
        self.net = model
        self.task_manager = task_manager

        self.ce = torch.nn.functional.cross_entropy
        self.opt = optim.Adam(self.net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        self.margin = args.gem_args['memory_strength']
        self.n_memories = args.gem_args['n_memories']

        # allocate episodic memory
        # for semi-supervised data, it will store the training mask for every old tasks
        self.memory_data = []
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), len(args.task_seq)).cuda()
        
        # allocate counters
        self.observed_tasks = []
        self.current_task = -1
        self.mem_cnt = 0
    
    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            
        # Update ring buffer storing examples from current task        
        if t >= len(self.memory_data):
            tmask = train_ids[0:self.n_memories]
            self.memory_data.append(tmask)
        
        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:

            if args.cross_task_edges:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, args.task_seq[old_task_i]),'rb'))
            else:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                              args.task_seq[old_task_i]), 'rb'))

            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features_, labels_ = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            self.net.zero_grad()

            offset1, offset2 = self.task_manager.get_label_offset(old_task_i)
            output, _ = self.net(subgraph, features_)

            old_task_loss = self.ce(
                output[self.memory_data[old_task_i], offset1: offset2],
                labels_[self.memory_data[old_task_i]])# - offset1)

            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims,
                            old_task_i)

        # now compute the grad on the current minibatch
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)

        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t],
                               self.grad_dims)
        
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t

        # Update ring buffer storing examples from current task
        if t >= len(self.memory_data):
            tmask = random.choices(train_ids, k=self.n_memories) #train_ids[0:self.n_memories]
            self.memory_data.append(tmask)

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            if args.inter_task_edges:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                              args.task_seq[
                                                                                                  old_task_i]), 'rb'))
            else:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                                 args.task_seq[
                                                                                                     old_task_i]),'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features_, labels_ = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            self.net.zero_grad()

            offset1, offset2 = self.task_manager.get_label_offset(old_task_i-1)[1], self.task_manager.get_label_offset(old_task_i)[1]
            output, _ = self.net(subgraph, features_)
            output_labels_ = labels_[self.memory_data[old_task_i]]

            # balance the loss of data from different classes
            if args.cls_balance:
                n_per_cls = [(output_labels_ == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

            old_task_loss = self.ce(output[self.memory_data[old_task_i], offset1: offset2], output_labels_-offset1, weight=loss_w_[offset1: offset2])

            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, old_task_i)

        # now compute the grad on the current minibatch
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]-offset1
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t], self.grad_dims)
        self.opt.step()