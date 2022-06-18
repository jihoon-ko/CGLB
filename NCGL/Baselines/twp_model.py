import torch
from torch.autograd import Variable
import numpy as np

class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        # setup network
        self.net = model
        self.net.twp=True
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup losses
        self.ce = torch.nn.functional.cross_entropy
        # setup memories
        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.mem_mask = None
        # hyper-parameters
        self.lambda_l = args.twp_args['lambda_l']
        self.lambda_t = args.twp_args['lambda_t']
        self.beta = args.twp_args['beta']
        self.epochs = 0

    def forward(self, features):
        output, elist = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, elist = self.net(g, features)
        output_labels = labels[train_ids]
        # loss = self.ce((output[train_mask, : ]), labels[train_mask] - offset1)
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

        loss.backward(retain_graph=True)
        grad_norm = 0
        ps = list(self.net.parameters())
        for p in self.net.parameters():
            pg = p.grad.data.clone()
            # pg = p.grad.clone()
            grad_norm += torch.norm(pg, p=1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            output, elist = self.net(g, features)
            # loss = self.ce((output[self.mem_mask, offset1: offset2]), labels[self.mem_mask] - offset1)
            if args.classifier_increase:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output[train_ids], output_labels, weight=loss_w_)
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(pg)

            eloss = torch.norm(elist[0])
            #print('eloss is {}'.format(eloss))
            eloss.backward()
            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[t].append(pg)

            self.current_task = t

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        ############### try to simulate the original setting of twp
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, elist = self.net(g, features)
        #labels = labels - offset1
        output_labels = labels[train_ids]
        # loss = self.ce((output[train_mask, : ]), labels[train_mask] - offset1)
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output_labels = output_labels - offset1
        loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

        loss.backward(retain_graph=True)
        grad_norm = 0

        # comment the following unfunctioning code to save memory
        '''
        for p in self.net.parameters():
            pg = p.grad.data.clone()
            # pg = p.grad.clone()
            grad_norm += torch.norm(pg, p=1)
        '''

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            output, elist = self.net(g, features)

            # loss = self.ce((output[self.mem_mask, offset1: offset2]), labels[self.mem_mask] - offset1)
            if args.classifier_increase:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(pg)

            eloss = torch.norm(elist[0])
            #print('eloss is {}'.format(eloss))
            eloss.backward()
            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[t].append(pg)

            self.current_task = t

