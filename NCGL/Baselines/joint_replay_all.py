import torch
import copy
from .ergnn_utils import *

samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True), 'MF':MF_sampler(plus=False), 'MF_plus':MF_sampler(plus=True)}
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
        self.bce = torch.nn.functional.binary_cross_entropy_with_logits

        # setup memories
        self.current_task = -1
        self.observed_masks = []
        self.aux_mask = []
        self.buffer_node_ids = []
        self.budget = args.ergnn_args['budget']
        self.d_CM = args.ergnn_args['d'] # d for CM sampler of ERGNN

    def forward(self, features):
        output = self.net(features)
        return output


    def observe(self, args, gs, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for g,features, labels_all, train_ids, ids_per_cls in zip(gs, featuress, labelss, train_idss, ids_per_clss):
            labels = labels_all[train_ids]
            output, _ = self.net(g, features)
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            if args.classifier_increase:
                loss += self.ce(output[train_ids, offset1:offset2], labels, weight=loss_w_[offset1: offset2])
            else:
                loss += self.ce(output[train_ids], labels, weight=loss_w_)
        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, gs, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0

        for tt, (g,features, labels_all, train_ids, ids_per_cls) in enumerate(zip(gs, featuress, labelss, train_idss, ids_per_clss)):
            labels = labels_all[train_ids]
            offset1, offset2 = self.task_manager.get_label_offset(tt - 1)[1], self.task_manager.get_label_offset(tt)[1]
            output, _ = self.net(g, features)
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids, offset1:offset2], labels-offset1, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, gs, dataloader, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions, output_labels, weight=loss_w_)
            loss.backward()
            self.opt.step()

    def observe_class_IL_batch(self, args, gs, dataloader, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:,offset1:offset2], output_labels, weight=loss_w_[offset1:offset2])
            loss.backward()
            self.opt.step()

    def observe_task_IL_crsedge(self, args, g, features, labels_all, t, train_ids, ids_per_cls_all, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        cls_retain = []
        for clss in args.task_seq:
            cls_retain.extend(clss)
        output, _ = self.net(g, features)

        for tt,task in enumerate(args.task_seq[0:t+1]):
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in task]
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
            train_ids_current_task = []
            for ids in ids_per_cls_train:
                train_ids_current_task.extend(ids)
            labels = labels_all[train_ids_current_task]
            offset1, offset2 = self.task_manager.get_label_offset(tt - 1)[1], self.task_manager.get_label_offset(tt)[1]
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids_current_task, offset1:offset2], labels - offset1, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_class_IL_crsedge(self, args, g, features, labels_all, t, train_ids, ids_per_cls_all, dataset):
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        cls_retain = []
        for clss in args.task_seq:
            cls_retain.extend(clss)
        output, _ = self.net(g, features)
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for tt,task in enumerate(args.task_seq[0:t+1]):
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in task]
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
            train_ids_current_task = []
            for ids in ids_per_cls_train:
                train_ids_current_task.extend(ids)
            labels = labels_all[train_ids_current_task]

            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids_current_task, offset1:offset2], labels, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()