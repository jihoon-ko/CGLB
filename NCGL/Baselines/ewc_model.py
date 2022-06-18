import torch


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()
        self.reg = args.ewc_args['memory_strength']

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.mem_mask = None
        self.epochs = 0

        #self.n_memories = args.n_memories

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output = self.net(g, features)
        output_labels = labels[train_ids]
        if isinstance(output,tuple):
            output = output[0]

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

        #loss = self.ce((output[train_mask, offset1: offset2]), labels[train_mask] - offset1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        # if new task
        if last_epoch == 0:
            self.net.zero_grad()
            #offset1, offset2 = self.task_manager.get_label_offset(self.current_task)
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            # self.ce((output[self.mem_mask, offset1: offset2]), labels[self.mem_mask] - offset1).backward()
            if args.classifier_increase:
                self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2]).backward()
            else:
                self.ce(output[train_ids], output_labels, weight=loss_w_).backward()

            self.fisher[t] = []
            self.optpar[t] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.fisher[t].append(pg)
                self.optpar[t].append(pd)
            self.current_task = t

        return loss

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output = self.net(g, features)
        labels = labels - offset1
        output_labels = labels[train_ids]
        if isinstance(output,tuple):
            output = output[0]

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

        #loss = self.ce((output[train_mask, offset1: offset2]), labels[train_mask] - offset1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        # if new task
        if last_epoch == 0:
            self.net.zero_grad()
            #offset1, offset2 = self.task_manager.get_label_offset(self.current_task)
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            # self.ce((output[self.mem_mask, offset1: offset2]), labels[self.mem_mask] - offset1).backward()
            if args.classifier_increase:
                self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2]).backward()
            else:
                self.ce(output[train_ids], output_labels, weight=loss_w_).backward()

            self.fisher[t] = []
            self.optpar[t] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.fisher[t].append(pg)
                self.optpar[t].append(pd)
            self.current_task = t

        return loss

