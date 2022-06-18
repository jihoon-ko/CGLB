import torch
import copy
from .ergnn_utils import *
import pickle

samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True), 'MF':MF_sampler(plus=False), 'MF_plus':MF_sampler(plus=True),'random':random_sampler(plus=False)}
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.ergnn_args['sampler']]

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = args.ergnn_args['budget']
        self.d_CM = args.ergnn_args['d'] # d for CM sampler of ERGNN
        self.aux_g = None

    def forward(self, features):
        output = self.net(features)
        return output
    
    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size/(buffer_size+n_nodes)

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
            loss = self.ce(output[train_ids,offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], labels[train_ids], weight=loss_w_)

        if t!=self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h, self.d_CM)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            if t>0:
                g, self.ids_per_cls, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                self.aux_g = g.to(device='cuda:{}'.format(features.get_device()))
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        if t!=0:
            output, _ = self.net(self.aux_g, self.aux_features)
            if args.classifier_increase:
                loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels, weight=self.aux_loss_w_[offset1: offset2])
            else:
                loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
            loss = beta*loss + (1-beta)*loss_aux

        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = 0
        for k in self.buffer_node_ids:
            buffer_size+=len(self.buffer_node_ids[k])
        beta = buffer_size/(buffer_size+n_nodes)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t!=self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h, self.d_CM)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            g, self.ids_per_cls, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
            self.aux_g.append(g.to(device='cuda:{}'.format(features.get_device())))
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            self.aux_loss_w_.append(loss_w_)

        if t!=0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][offset1: offset2])
                loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()

    def observe_batch(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size/(buffer_size+n_nodes)

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
            loss = self.ce(output[train_ids,offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], labels[train_ids], weight=loss_w_)

        if t!=self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls, self.budget, features, self.net.second_last_h, self.d_CM)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            if t>0:
                g, self.ids_per_cls, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                self.aux_g = g.to(device='cuda:{}'.format(features.get_device()))
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                self.aux_loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        if t!=0:
            output, _ = self.net(self.aux_g, self.aux_features)
            if args.classifier_increase:
                loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels, weight=self.aux_loss_w_[offset1: offset2])
            else:
                loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)
            loss = beta*loss + (1-beta)*loss_aux
        loss.backward()
        self.opt.step()


