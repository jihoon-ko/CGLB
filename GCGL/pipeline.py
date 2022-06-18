import numpy as np
import torch
import importlib
from dgllife.model import load_pretrained
from dgllife.utils import EarlyStopping, Meter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import copy
from utils import collate_molgraphs, load_model, load_twpmodel, GraphLevelDataset
import os
import errno
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def predict(args, model, bg, task_i):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, edge_feats)
        else:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, edge_feats, task_i)
    else:
        if args['backbone'] in ['GCN', 'GAT', 'Weave']:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats)
        else:
            return model(bg.to(f"cuda:{args['gpu']}"), node_feats, task_i)

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer, task_i):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.cuda(), masks.cuda()
        logits = predict(args, model, bg, task_i)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Mask non-existing labels
        loss = loss_criterion(logits, labels) * (masks != 0).float()
        loss = loss[:,task_i].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)


def run_an_eval_epoch(args, model, data_loader, task_i):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.cuda()
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            eval_meter.update(logits, labels, masks)

    return eval_meter.compute_metric(args['metric_name'])[task_i]


def run_an_eval_epoch_multiclass(args, model, data_loader, task_i):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        ids_per_cls = [(y_true==i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
        acc_per_cls = [(y_pred[ids]==y_true[ids]).sum()/len(ids) for ids in ids_per_cls]
    return sum(acc_per_cls).item()/len(acc_per_cls)


def run_eval_epoch(args, model, data_loader, task_i):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        if args['classifier_increase']:
            t_end = task_i + 1
        else:
            t_end = args['n_tasks']
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg_, labels, masks = batch_data
            labels = labels.cuda()
            logits = torch.tensor([]).cuda(args['gpu'])
            #logitss = []
            for t in range(t_end):
                bg = copy.deepcopy(bg_)
                logits_ = predict(args, model, bg, t)
                if isinstance(logits_, tuple):
                    logits_ = logits_[0]
                logits = torch.cat([logits,logits_[:,t].view(-1,1)], dim=-1)
            if isinstance(logits, tuple):
                logits = logits[0]
            eval_meter.update(logits, labels[:,0:t_end], masks)

        test_score =  eval_meter.compute_metric(args['metric_name'])
        score_mean = round(np.mean(test_score),4)

        for t in range(t_end):
            score = test_score[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return test_score

def run_eval_epoch_multiclass(args, model, data_loaders, task_i):
    model.eval()
    acc_learnt_tsk = np.zeros(args['n_tasks'])
    with torch.no_grad():
        if args['clsIL']:
            t_end = task_i + 1
        else:
            t_end = args['n_tasks']
        for tid, data_loader in enumerate(data_loaders):
            y_pred = []
            y_true = []
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                logits = predict(args, model, bg, task_i)[:,args['tasks'][tid]]
                if isinstance(logits, tuple):
                    logits = logits[0]
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
            acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
            acc_learnt_tsk[tid] = sum(acc_per_cls).item()/len(acc_per_cls)

        score_mean = round(np.mean(acc_learnt_tsk),4)

        for t in range(t_end):
            score = acc_learnt_tsk[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return acc_learnt_tsk

def run_an_eval_epoch_multiclass_tskIL(args, model, data_loader, task_i):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, args['tasks'][task_i]]
            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.argmax(-1)
        y_true = torch.cat(y_true, dim=0)
        for i, c in enumerate(args['tasks'][task_i]):
            y_true[y_true == c] = i
        ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
        acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
    return sum(acc_per_cls).item() / len(acc_per_cls)

def run_eval_epoch_multiclass_tskIL(args, model, data_loaders, task_i):
    model.eval()
    acc_learnt_tsk = np.zeros(args['n_tasks'])
    with torch.no_grad():
        if args['clsIL']:
            t_end = task_i + 1
        else:
            t_end = args['n_tasks']
        for tid, data_loader in enumerate(data_loaders):
            y_pred = []
            y_true = []
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                logits = predict(args, model, bg, task_i)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, args['tasks'][tid]]
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.argmax(-1)
            y_true = torch.cat(y_true, dim=0)
            for i, c in enumerate(args['tasks'][tid]):
                y_true[y_true == c] = i
            ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
            acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
            acc_learnt_tsk[tid] = sum(acc_per_cls).item() / len(acc_per_cls)

        score_mean = round(np.mean(acc_learnt_tsk[0:t_end]), 4)

        for t in range(t_end):
            score = acc_learnt_tsk[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return acc_learnt_tsk

def run_an_eval_epoch_multiclass_clsIL(args, model, data_loader, task_i):
    model.eval()
    y_pred = []
    y_true = []
    learnt_cls = []
    for tid in range(task_i+1):
        learnt_cls.extend(args['tasks'][tid])
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            logits = predict(args, model, bg, task_i)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, learnt_cls]
            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.argmax(-1)
        y_true = torch.cat(y_true, dim=0)
        for i, c in enumerate(learnt_cls):
            y_true[y_true == c] = i
        ids_per_cls = [(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()]
        acc_per_cls = [(y_pred[ids] == y_true[ids]).sum() / len(ids) for ids in ids_per_cls]
    return sum(acc_per_cls).item() / len(acc_per_cls)

def run_eval_epoch_multiclass_clsIL(args, model, data_loaders, task_i):
    model.eval()
    acc_learnt_tsk = np.zeros(args['n_tasks'])
    learnt_cls,allclss = [],[]
    for tid in range(task_i + 1):
        learnt_cls.extend(args['tasks'][tid])
    for tid in args['tasks']:
        allclss.extend(tid)
    with torch.no_grad():
        selected_clss = allclss if args['method'] is 'jointtrain' else learnt_cls
        t_end = task_i + 1 if args['method'] is not 'jointtrain' else args['n_tasks']
        for tid, data_loader in enumerate(data_loaders[0:t_end]):
            y_pred,y_true = [],[]
            for batch_id, batch_data in enumerate(data_loader):
                smiles, bg, labels, masks = batch_data
                logits = predict(args, model, bg, task_i)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, selected_clss]
                y_pred.append(logits.detach().cpu())
                y_true.append(labels.detach().cpu())
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = y_pred.argmax(-1)
            y_true = torch.cat(y_true, dim=0)
            for i, c in enumerate(selected_clss):
                y_true[y_true == c] = i
            ids_per_cls = {i:(y_true == i).nonzero().view(-1).tolist() for i in y_true.int().unique().tolist()}
            acc_per_cls = [(y_pred[ids_per_cls[ids]] == y_true[ids_per_cls[ids]]).sum().item() / len(ids_per_cls[ids]) for ids in ids_per_cls]
            if args['method'] is 'jointtrain':
                acc_learnt_tsk[tid] = sum(acc_per_cls) / len(acc_per_cls)
            acc_learnt_tsk[tid] = sum(acc_per_cls) / len(acc_per_cls)

        score_mean = round(np.mean(acc_learnt_tsk[0:t_end]), 4)

        for t in range(t_end):
            score = acc_learnt_tsk[t]
            print(f"T{t:02d} {score:.4f}|", end="")

        print(f"score_mean: {score_mean}", end="")
        print()

    return acc_learnt_tsk



def multi_label(args):
    torch.cuda.set_device(args['gpu'])
    G = GraphLevelDataset(args)
    dataset, train_set, val_set, test_set = G.dataset, G.train_set, G.val_set, G.test_set
    if args['dataset'] in ['PubChemBioAssayAromaticity','MiniGCDataset']:
        args['n_cls'] = dataset.labels.max().int().item()
        train_loader_ = [DataLoader(s, batch_size=args['batch_size'],
                                   collate_fn=collate_molgraphs, shuffle=True) for s in train_set]
        val_loader_ = [DataLoader(s, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs) for s in val_set]
        test_loader = [DataLoader(s, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs) for s in test_set]
    else:
        args['n_cls'] = dataset.labels.shape[1]

        args['tasks'] = [list(range(i, i + args['n_cls_per_task'])) for i in
                         range(0, args['n_cls'], args['n_cls_per_task'])]
        train_loader_ = DataLoader(train_set, batch_size=args['batch_size'],
                                   collate_fn=collate_molgraphs, shuffle=True)
        val_loader_ = DataLoader(val_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
        test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
    args['d_data'] = dataset.graphs[0].ndata['h'].shape[1]


    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = len(args['tasks']) #dataset.n_tasks
        args['n_outheads'] = args['n_tasks']
        if args['method'] == 'twp':
            model = load_twpmodel(args)
            print(model)
        else:
            model = load_model(args)
            for name, parameters in model.named_parameters():
                print(name, ':', parameters.size())
        
        method = args['method']
        life_model = importlib.import_module(f'LifeModel.{method}_model')
        life_model_ins = life_model.NET(model, args)
        data_loader = DataLoader(train_set, batch_size=len(train_set),
            collate_fn=collate_molgraphs, shuffle=True)
        life_model_ins.data_loader = data_loader

        if args['dataset'] in ['PubChemBioAssayAromaticity']:
            loss_criterion = torch.nn.functional.cross_entropy
        else:
            loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights(train_set.indices).cuda(),
                                               reduction='none')

    model.cuda()
    score_mean = []
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    for tid,task_i in enumerate(args['tasks']):
        print('\n********'+ str([tid,task_i]))
        if args['dataset'] in ['PubChemBioAssayAromaticity']:
            train_loader,val_loader = train_loader_[tid], val_loader_[tid]
            val_func, test_func = run_an_eval_epoch_multiclass, run_eval_epoch_multiclass
        else:
            train_loader, val_loader = train_loader_, val_loader_
            val_func, test_func = run_an_eval_epoch, run_eval_epoch
        stopper = EarlyStopping(patience=args['patience'])
        for epoch in range(args['num_epochs']):
            # Train
            if args['method'] == 'lwf':
                life_model_ins.observe(train_loader, loss_criterion, tid, args, prev_model)
            else:
                life_model_ins.observe(train_loader, loss_criterion, tid, args)

            # Validation and early stop
            val_score = val_func(args, model, val_loader, tid)
            early_stop = stopper.step(val_score, model)
            
            if early_stop:
                print(epoch)
                break

        if not args['pre_trained']:
            stopper.load_checkpoint(model)

        score_matrix[tid] = test_func(args, model, test_loader, tid)
        prev_model = copy.deepcopy(life_model_ins).cuda()

    AP = round(np.mean(score_matrix[-1,:]),4)
    print('AP: ', round(np.mean(score_matrix[-1,:]),4))
    backward = []
    for t in range(args['n_tasks']-1):
        b = score_matrix[args['n_tasks']-1][t]-score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward),4)        
    print('AF: ', mean_backward)
    return AP, mean_backward, score_matrix
    

def get_pipeline(args):
    if args['clsIL']:
        if args['method'] in ['joint', 'Joint', 'joint_replay_all', 'jointreplay']:
            #return pipeline_class_IL_joint
            return pipeline_task_IL_multi_class
        else:
            #return pipeline_class_IL
            return pipeline_task_IL_multi_class
    else:
        if args['dataset'] in ['SIDER-tIL', 'Tox21-tIL']:
            return pipeline_task_IL_multi_label
        else:
            if args['method'] in ['joint', 'Joint', 'joint_replay_all']:
                #return pipeline_task_IL_multi_class_joint
                return pipeline_task_IL_multi_class
            else:
                return pipeline_task_IL_multi_class


def pipeline_task_IL_multi_label(args):
    torch.cuda.set_device(args['gpu'])
    # set_random_seed(args['random_seed'])
    G = GraphLevelDataset(args)
    dataset, train_set, val_set, test_set = G.dataset, G.train_set, G.val_set, G.test_set
    if args['dataset'] in ['PubChemBioAssayAromaticity']:
        args['n_cls'] = dataset.labels.max().int().item()
        train_loader_ = [DataLoader(s, batch_size=args['batch_size'],
                                    collate_fn=collate_molgraphs, shuffle=True) for s in train_set]
        val_loader_ = [DataLoader(s, batch_size=args['batch_size'],
                                  collate_fn=collate_molgraphs) for s in val_set]
        test_loader = [DataLoader(s, batch_size=args['batch_size'],
                                  collate_fn=collate_molgraphs) for s in test_set]
    else:
        args['n_cls'] = dataset.labels.shape[1]

        args['tasks'] = [list(range(i, i + args['n_cls_per_task'])) for i in
                         range(0, args['n_cls'], args['n_cls_per_task'])]
        train_loader_ = DataLoader(train_set, batch_size=args['batch_size'],
                                   collate_fn=collate_molgraphs, shuffle=True)
        val_loader_ = DataLoader(val_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
        test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs)
    args['d_data'] = dataset.graphs[0].ndata['h'].shape[1]


    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = len(args['tasks'])
        args['n_outheads'] = args['n_tasks']
        if args['method'] == 'twp':
            model = load_twpmodel(args)
            print(model)
        else:
            model = load_model(args)
            for name, parameters in model.named_parameters():
                print(name, ':', parameters.size())

        method = args['method']
        life_model = importlib.import_module(f'Baselines.{method}_model')
        life_model_ins = life_model.NET(model, args)
        data_loader = DataLoader(train_set, batch_size=len(train_set),
                                 collate_fn=collate_molgraphs, shuffle=True)
        life_model_ins.data_loader = data_loader

        if args['dataset'] in ['PubChemBioAssayAromaticity']:
            loss_criterion = torch.nn.functional.cross_entropy
        else:
            loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights(train_set.indices).cuda(),
                                               reduction='none')

    model.cuda()
    score_mean = []
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    for tid, task_i in enumerate(args['tasks']):
        print('\n********' + str([tid, task_i]))
        if args['dataset'] in ['PubChemBioAssayAromaticity']:
            train_loader, val_loader = train_loader_[tid], val_loader_[tid]
            val_func, test_func = run_an_eval_epoch_multiclass, run_eval_epoch_multiclass
        else:
            train_loader, val_loader = train_loader_, val_loader_
            val_func, test_func = run_an_eval_epoch, run_eval_epoch
        # loss_criterion = BCEWithLogitsLoss(pos_weight=dataset.task_pos_weights(train_set.indices)[0:task_i+1].cuda(), reduction='none')
        stopper = EarlyStopping(patience=args['patience'])
        for epoch in range(args['num_epochs']):
            # Train
            if args['method'] == 'lwf':
                life_model_ins.observe(train_loader, loss_criterion, tid, args, prev_model)
            else:
                life_model_ins.observe(train_loader, loss_criterion, tid, args)

            # Validation and early stop
            val_score = val_func(args, model, val_loader, tid)
            early_stop = stopper.step(val_score, model)

            if early_stop:
                print(epoch)
                break

        if not args['pre_trained']:
            stopper.load_checkpoint(model)

        score_matrix[tid] = test_func(args, model, test_loader, tid)
        prev_model = copy.deepcopy(life_model_ins).cuda()

    AP = round(np.mean(score_matrix[-1, :]), 4)
    print('AP: ', round(np.mean(score_matrix[-1, :]), 4))
    backward = []
    for t in range(args['n_tasks'] - 1):
        b = score_matrix[args['n_tasks'] - 1][t] - score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward), 4)
    print('AF: ', mean_backward)
    return AP, mean_backward, score_matrix

def pipeline_task_IL_multi_class(args):
    torch.cuda.set_device(args['gpu'])
    G = GraphLevelDataset(args)
    dataset, train_set, val_set, test_set = G.dataset, G.train_set, G.val_set, G.test_set
    coll_f = collate_molgraphs
    args['n_cls'] = dataset.labels.max().int().item() + 1
    args['n_outheads'] = args['n_cls']

    train_loader = [DataLoader(s, batch_size=args['batch_size'], collate_fn=coll_f, shuffle=True) for s in train_set]
    val_loader = [DataLoader(s, batch_size=args['batch_size'], collate_fn=coll_f) for s in val_set]
    test_loader = [DataLoader(s, batch_size=args['batch_size'],collate_fn=coll_f) for s in test_set]
    args['d_data'] = dataset.graphs[0].ndata['h'].shape[1]

    if args['pre_trained']:
        args['num_epochs'] = 0
        model = load_pretrained(args['exp'])
    else:
        args['n_tasks'] = len(args['tasks'])  # dataset.n_tasks
        if args['method'] == 'twp':
            model = load_twpmodel(args)
            print(model)
        else:
            model = load_model(args)
            for name, parameters in model.named_parameters():
                print(name, ':', parameters.size())

        method = args['method']
        life_model = importlib.import_module(f'Baselines.{method}_model')
        life_model_ins = life_model.NET(model, args)
        data_loader = [DataLoader(s, batch_size=len(s), collate_fn=coll_f, shuffle=True) for s in train_set]
        life_model_ins.data_loader = data_loader

        loss_criterion = torch.nn.functional.cross_entropy

    model.cuda()
    score_matrix = np.zeros([args['n_tasks'], args['n_tasks']])

    prev_model = None
    for tid, task_i in enumerate(args['tasks']):
        print('\n********' + str([tid, task_i]),{i:args['n_per_cls'][i] for i in task_i})
        if args['clsIL']:
            train_func,val_func, test_func = life_model_ins.observe_clsIL,run_an_eval_epoch_multiclass_clsIL, run_eval_epoch_multiclass_clsIL
        elif not args['clsIL']:
            train_func,val_func, test_func = life_model_ins.observe_tskIL_multicls,run_an_eval_epoch_multiclass_tskIL, run_eval_epoch_multiclass_tskIL
        stopper = EarlyStopping(patience=args['patience'])
        for epoch in range(args['num_epochs']):
            # Train
            if args['method'] == 'lwf':
                train_func(train_loader, loss_criterion, tid, args, prev_model)
            else:
                train_func(train_loader, loss_criterion, tid, args)

        score_matrix[tid] = test_func(args, model, test_loader, tid)
        prev_model = copy.deepcopy(life_model_ins).cuda()

    AP = round(np.mean(score_matrix[-1, :]), 4)
    print('AP: ', AP)
    backward = []
    for t in range(args['n_tasks'] - 1):
        b = score_matrix[args['n_tasks'] - 1][t] - score_matrix[t][t]
        backward.append(round(b, 4))
    mean_backward = round(np.mean(backward), 4)
    print('AF: ', mean_backward)
    return AP, mean_backward, score_matrix

