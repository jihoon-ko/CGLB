import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluate, NodeLevelDataset, evaluate_batch
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl

def get_pipeline(args):
    if args.minibatch:
        if args.ILmode == 'classIL':
            if args.inter_task_edges:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_class_IL_inter_edge_minibatch_joint
                else:
                    return pipeline_class_IL_inter_edge_minibatch
            else:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_task_IL_no_inter_edge_minibatch_joint # for Joint, the pipeline is same for both task-IL and class-IL
                else:
                    return pipeline_class_IL_no_inter_edge_minibatch
        elif args.ILmode == 'taskIL':
            if args.inter_task_edges:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_task_IL_inter_edge_minibatch_joint
                else:
                    return pipeline_task_IL_inter_edge_minibatch
            else:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_task_IL_no_inter_edge_minibatch_joint
                else:
                    return pipeline_task_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            if args.inter_task_edges:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_class_IL_inter_edge_joint
                else:
                    return pipeline_class_IL_inter_edge
            else:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_class_IL_no_inter_edge_joint
                else:
                    return pipeline_class_IL_no_inter_edge
        elif args.ILmode == 'taskIL':
            if args.inter_task_edges:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_task_IL_inter_edge_joint
                else:
                    return pipeline_task_IL_inter_edge
            else:
                if args.method in ['joint', 'Joint', 'joint_replay_all']:
                    return pipeline_task_IL_no_inter_edge_joint
                else:
                    return pipeline_task_IL_no_inter_edge


def data_prepare(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.inter_task_edges:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),'rb'))
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'rb'))

        except:
            print(f'preparing data for task {task}')
            mkdir_if_missing('./data/inter_tsk_edge')
            mkdir_if_missing('./data/no_inter_tsk_edge')
            if args.inter_task_edges:
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                mkdir_if_missing('./data/inter_tsk_edge')
                with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
                '''
                subgraph, ids_per_cls_, [train_ids_, valid_ids_, test_ids_] = dataset.get_graph(
                    tasks_to_retain=cls_retain)
                cls_ids_new = [cls_retain.index(i) for i in task_cls]
                ids_per_cls = [ids_per_cls_[i] for i in cls_ids_new]
                ids_per_cls_train = [list(set(ids).intersection(set(train_ids_))) for ids in ids_per_cls]
                ids_per_cls_val = [list(set(ids).intersection(set(valid_ids_))) for ids in ids_per_cls]
                ids_per_cls_test = [list(set(ids).intersection(set(test_ids_))) for ids in ids_per_cls]
                train_ids, valid_ids, test_ids = [], [], []
                for ids in ids_per_cls_train:
                    train_ids.extend(ids)
                for ids in ids_per_cls_val:
                    valid_ids.extend(ids)
                for ids in ids_per_cls_test:
                    test_ids.extend(ids)
                with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)
                '''
            else:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)

def pipeline_task_IL_no_inter_edge(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    data_prepare(args)
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))

        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model, train_ids,
                                               ids_per_cls, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)

        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,args.task_seq[t]),'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
            './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_task_IL(args, subgraph, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        acc_mean = []
        # test
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_no_inter_edge_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, args.task_seq[t]), 'rb'))
            '''
            try:
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,args.task_seq[t]), 'rb'))
            except:
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                mkdir_if_missing('./data/no_inter_tsk_edge')
                with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),
                          'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_idx, test_ids]], f)
            '''
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(args.epochs):
            life_model_ins.observe_task_IL(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                         args.task_seq[t]), 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    cls_retain = []
    for clss in args.task_seq:
        cls_retain.extend(clss)
    for task, task_cls in enumerate(args.task_seq):
        cls_retain = []
        for clss in args.task_seq[0:task+1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(
            open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,args.task_seq[-1]),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        subgraphs = subgraph
        featuress = features
        labelss = labels
        train_idss = train_ids
        ids_per_clss_all = ids_per_cls_all

        for epoch in range(args.epochs):
            life_model_ins.observe_task_IL_crsedge(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss_all, dataset)

        acc_mean = []
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            #cls_ids_new = args.task_seq[t]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels_current_task = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'rb'))
        except:
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
            mkdir_if_missing('./data/no_inter_tsk_edge')
            with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                      'wb') as f:
                pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)
        '''
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1, label_offset2 = task_manager.get_label_offset(task)

        # training
        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model, train_ids, ids_per_cls, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()

        acc_mean = []
        # test
        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                            args.task_seq[t]), 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, args.n_cls, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)

            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")

        accs = acc_mean[:task+1]
        meana = round(np.mean(accs)*100,2)
        meanas.append(meana)
        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
            './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe(args, subgraph, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe(args, subgraph, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        acc_mean = []
        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            #label_offset1, label_offset2 = task_manager.get_label_offset(t)
            #labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                            args.task_seq[t]), 'rb'))
            '''
            try:
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                             args.task_seq[t]), 'rb'))
            except:
                mkdir_if_missing('./data/no_inter_tsk_edge')
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                                   task_cls),
                          'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_idx, test_ids]], f)
            '''
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        for epoch in range(args.epochs):
            life_model_ins.observe(args, subgraphs, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                         args.task_seq[t]), 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)

    for task, task_cls in enumerate(args.task_seq):
        cls_retain = []
        for clss in args.task_seq[:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(
            open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'rb'))
        except:
            mkdir_if_missing('./data/inter_tsk_edge')
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        for epoch in range(args.epochs):
            life_model_ins.observe_class_IL_crsedge(args, subgraph, features, labels, task, train_ids, ids_per_cls_all, dataset)

        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            cls_ids_new = args.task_seq[t]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            if args.classifier_increase:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate(model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_no_inter_edge_minibatch(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                 'rb'))
        '''
        try:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
                open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                     'rb'))
        except:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
            with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                      'wb') as f:
                pickle.dump([subgraph, ids_per_cls, [train_ids, valid_idx, test_ids]], f)
        '''
        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        #label_offset1, label_offset2 = task_manager.get_label_offset(task)

        # build the dataloader for mini batch training
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            #print(f'epoch {epoch}')
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls,
                                       dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()

        acc_mean = []
        # test
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                         args.task_seq[t]), 'rb'))
            #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            acc = evaluate_batch(args,model, subgraph, features, labels-label_offset1, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    if args.method == 'TEM':
        print('buffer_size', life_model_ins.SSEM_labels.shape)

    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_no_inter_edge_minibatch_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
        subgraphs, featuress, labelss, train_idss, ids_per_clss = [], [], [], [], []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                            args.task_seq[t]), 'rb'))
            '''
            try:
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(
                    './data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                             args.task_seq[t]), 'rb'))
            except:
                subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                mkdir_if_missing('./data/no_inter_tsk_edge')
                with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                                   task_cls),
                          'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_idx, test_ids]], f)
            '''
            #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            subgraphs.append(subgraph)
            featuress.append(features)
            labelss.append(labels)
            train_idss.append(train_ids)
            ids_per_clss.append(ids_per_cls)

        # build the dataloader for mini batch training
        train_ids_entire = []
        n_existing_nodes = 0
        for ids, g in zip(train_idss, subgraphs):
            new_ids = [i + n_existing_nodes for i in ids]
            train_ids_entire.extend(new_ids)
            n_existing_nodes += g.num_nodes()

        graph_entire = dgl.batch(subgraphs)
        dataloader = dgl.dataloading.NodeDataLoader(graph_entire, train_ids_entire, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            life_model_ins.observe_task_IL_batch(args, subgraphs, dataloader, featuress, labelss, task, train_idss, ids_per_clss, dataset)

        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,
                                                                                         args.task_seq[t]), 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_no_inter_edge_minibatch(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
            open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                 'rb'))
        '''
        try:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(
                open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                     'rb'))
        except:
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
            with open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls),
                      'wb') as f:
                pickle.dump([subgraph, ids_per_cls, [train_ids, valid_idx, test_ids]], f)
        '''
        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        #label_offset1, label_offset2 = task_manager.get_label_offset(task)

        # build the dataloader for mini batch training
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model, train_ids, ids_per_cls,
                                       dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls, dataset)
                torch.cuda.empty_cache()  # tracemalloc.stop()

        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        # test
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open('./data/no_inter_tsk_edge/{}_{}.pkl'.format(args.dataset,args.task_seq[t]), 'rb'))
            #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_minibatch(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
            './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]

        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)
        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids_current_task, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
                torch.cuda.empty_cache()
        acc_mean = []
        # test
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_task_IL_inter_edge_minibatch_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    for task, task_cls in enumerate(args.task_seq):
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(
            open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,args.task_seq[-1]),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            life_model_ins.observe_task_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_all, dataset)

        acc_mean = []
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            cls_ids_new = args.task_seq[t]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            label_offset1, label_offset2 = task_manager.get_label_offset(t - 1)[1], task_manager.get_label_offset(t)[1]
            labels_current_task = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels_current_task, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_minibatch(args):
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    prev_model = None
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
            './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = pickle.load(open(
                './data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)

        cls_ids_new = [cls_retain.index(i) for i in task_cls]
        ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]

        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
        train_ids_current_task = []
        for ids in ids_per_cls_train:
            train_ids_current_task.extend(ids)

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids_current_task, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            if args.method == 'lwf':
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, prev_model,
                                               train_ids_current_task, ids_per_cls_current_task, dataset)
            else:
                life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids_current_task,
                                               ids_per_cls_current_task, dataset)
        acc_mean = []
        # test
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t in range(task + 1):
            cls_ids_new = [cls_retain.index(i) for i in args.task_seq[t]]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            #label_offset1, label_offset2 = task_manager.get_label_offset(t)
            #labels = labels - label_offset1
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()
        prev_model = copy.deepcopy(model).cuda()

    print('AP: ', acc_mean)
    backward = []
    forward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix

def pipeline_class_IL_inter_edge_minibatch_joint(args):
    args.method = 'joint_replay_all'
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)

    task_manager = semi_task_manager()

    model = get_model(dataset, args).cuda()
    life_model = importlib.import_module(f'Baselines.{args.method}')
    life_model_ins = life_model.NET(model, task_manager, args)

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    meanas = []
    n_cls_so_far = 0
    data_prepare(args)
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        task_manager.add_task(task, n_cls_so_far)
    '''
    cls_retain = []
    for clss in args.task_seq:
        cls_retain.extend(clss)
    '''
    for task, task_cls in enumerate(args.task_seq):
        cls_retain = []
        for clss in args.task_seq[0:task + 1]:
            cls_retain.extend(clss)
        subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(
            open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset, task_cls), 'rb'))
        '''
        try:
            subgraph, ids_per_cls_all, [train_ids, valid_idx, test_ids] = pickle.load(open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls), 'rb'))
        except:
            subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
            mkdir_if_missing('./data/inter_tsk_edge')
            with open('./data/inter_tsk_edge/{}_{}.pkl'.format(args.dataset,task_cls),'wb') as f:
                pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
        '''
        #subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        dataloader = dgl.dataloading.NodeDataLoader(subgraph, train_ids, args.nb_sampler,
                                                    batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                    drop_last=False)

        for epoch in range(args.epochs):
            life_model_ins.observe_class_IL_batch(args, subgraph, dataloader, features, labels, task, train_ids, ids_per_cls_all, dataset)

        acc_mean = []
        label_offset1, label_offset2 = task_manager.get_label_offset(task)
        for t, cls_ids_new in enumerate(args.task_seq[0:task+1]):
            cls_ids_new = args.task_seq[t]
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in cls_ids_new]
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls_current_task]
            if args.classifier_increase:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            else:
                acc = evaluate_batch(args,model, subgraph, features, labels, test_ids, label_offset1, label_offset2,
                               cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        accs = acc_mean[:task + 1]
        meana = round(np.mean(accs) * 100, 2)
        meanas.append(meana)

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean: {acc_mean}", end="")
        print()

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
