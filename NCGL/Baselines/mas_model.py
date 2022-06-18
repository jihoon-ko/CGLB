import torch


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()
        self.reg = args.mas_args['memory_strength']

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.n_seen_examples = 0
        self.mem_mask = None
        self.epochs = 0

        #self.n_memories = args.mas_args['n_memories']

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = features.shape[0]

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        #loss = self.ce((output[train_mask, offset1: offset2]), labels[train_mask] - offset1)
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

        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            #offset1, offset2 = self.task_manager.get_label_offset(self.current_task)
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            # output = output[self.mem_mask, offset1: offset2]
            if args.classifier_increase:
                output = output[train_ids, offset1:offset2]
            else:
                output = output[train_ids, :]

            output.pow_(2)
            loss = output.mean()
            self.net.zero_grad()
            loss.backward()

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                # self.fisher.append(pg)
                new_fisher.append(pg)
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]) / (
                                self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i] / n_new_examples)
                self.n_seen_examples = n_new_examples

            self.current_task = t

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()
        n_new_examples = features.shape[0]

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        #loss = self.ce((output[train_mask, offset1: offset2]), labels[train_mask] - offset1)
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t > 0:
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()

        if last_epoch==0:
            self.optpar = []
            new_fisher = []
            #offset1, offset2 = self.task_manager.get_label_offset(self.current_task)
            output = self.net(g, features)
            if isinstance(output, tuple):
                output = output[0]
            # output = output[self.mem_mask, offset1: offset2]
            output = output[train_ids, offset1:offset2]

            output.pow_(2)
            loss = output.mean()
            self.net.zero_grad()
            loss.backward()

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                # self.fisher.append(pg)
                new_fisher.append(pg)
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]) / (
                                self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i] / n_new_examples)
                self.n_seen_examples = n_new_examples

            self.current_task = t
