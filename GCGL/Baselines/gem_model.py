import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np
from .gem_utils import store_grad, overwrite_grad, project2cone2


def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])
        self.data_loader = None

        self.margin = args['memory_strength']
        self.n_memories = args['n_memories']

        # for semi-supervised data, it will store the training mask for every old tasks
        self.memory_data = []

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), args['n_tasks']).cuda()
        
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

        self.mask = [i for i in range(args['n_train_examples'])]

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):

        if task_i != self.old_task:
            self.observed_tasks.append(task_i)
            self.old_task = task_i
        
        # Update ring buffer storing examples from current task
        if task_i >= len(self.memory_data):
            tmask = np.random.choice(self.mask, self.n_memories, replace = False)
            tmask = np.array(tmask)
            self.memory_data.append(tmask)


        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            self.net.zero_grad()
            # fwd/bwd on the examples in the memory

            for batch_id, batch_data in enumerate(self.data_loader):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                loss = loss_criterion(logits, labels) * (masks != 0).float()
                old_task_loss = loss[:,old_task_i].mean()

                old_task_loss.backward()
                store_grad(self.net.parameters, self.grads, self.grad_dims,
                                old_task_i)

        self.net.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)
        
        train_score = np.mean(train_meter.compute_metric(args['metric_name']))

         # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            t = task_i
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
        
        self.optimizer.step()

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args):

        if task_i != self.old_task:
            self.observed_tasks.append(task_i)
            self.old_task = task_i

        # Update ring buffer storing examples from current task
        if task_i >= len(self.memory_data):
            tmask = np.random.choice(self.mask, min(self.n_memories,len(self.mask)), replace=False)
            tmask = np.array(tmask)
            self.memory_data.append(tmask)

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            self.net.zero_grad()
            # fwd/bwd on the examples in the memory
            clss = []
            for tid in range(old_task_i + 1):
                clss.extend(args['tasks'][tid])
            #clss = args['tasks'][old_task_i]
            for batch_id, batch_data in enumerate(self.data_loader[old_task_i]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(clss):
                    labels[labels == c] = i

                old_task_loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
                #old_task_loss = loss[:, old_task_i].mean()

                old_task_loss.backward()
                store_grad(self.net.parameters, self.grads, self.grad_dims,
                           old_task_i)

        self.net.train()
        #train_meter = Meter()
        clss = []
        for tid in range(task_i + 1):
            clss.extend(args['tasks'][tid])
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            t = task_i
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

        self.optimizer.step()

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args):

        if task_i != self.old_task:
            self.observed_tasks.append(task_i)
            self.old_task = task_i

        # Update ring buffer storing examples from current task
        if task_i >= len(self.memory_data):
            tmask = np.random.choice(self.mask, min(self.n_memories,len(self.mask)), replace=False)
            tmask = np.array(tmask)
            self.memory_data.append(tmask)

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            self.net.zero_grad()
            # fwd/bwd on the examples in the memory
            clss = args['tasks'][old_task_i]
            for batch_id, batch_data in enumerate(self.data_loader[old_task_i]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(clss):
                    labels[labels == c] = i

                old_task_loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
                #old_task_loss = loss[:, old_task_i].mean()

                old_task_loss.backward()
                store_grad(self.net.parameters, self.grads, self.grad_dims,
                           old_task_i)

        self.net.train()
        #train_meter = Meter()
        clss = args['tasks'][task_i]
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            t = task_i
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

        self.optimizer.step()
