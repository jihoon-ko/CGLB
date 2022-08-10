import os
import traceback
import argparse
from training.utils import set_seed, mkdir_if_missing
from distutils.util import strtobool
from pipeline import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGLB')
    parser.add_argument("--dataset", type=str, default='Products-CL', help='Products-CL, Reddit-CL, Arxiv-CL, CoraFull-CL')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs, default = 200")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--backbone', type=str, default='GCN',
                        help="backbone GNN, [GAT, GCN, GIN]")
    parser.add_argument('--method', type=str,
                        choices=["bare", 'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain', 'ergnn', 'joint','Joint'], default="bare",
                        help="baseline continual learning method")

    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=30, help='')

    # parameters for continual learning settings
    parser.add_argument('--share-labels', type=strtobool, default=False,
                        help='task-IL specific, whether to share output label space for different tasks')
    parser.add_argument('--inter-task-edges', type=strtobool, default=False,
                        help='whether to keep the edges connecting nodes from different tasks')
    parser.add_argument('--classifier-increase', type=strtobool, default=True,
                        help='class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False')

    # extra parameters
    parser.add_argument('--refresh_data', type=strtobool, default=False, help='whether to load existing splitting or regenerate')
    parser.add_argument('--d_dtat', default=None)
    parser.add_argument('--n_cls', default=None)
    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.4, 0.4], help='ratio of nodes used for valid and test')
    parser.add_argument('--transductive', type=strtobool, default=True, help='using transductive or inductive')
    parser.add_argument('--default_split', type=strtobool, default=False, help='whether to  use the data split provided by the dataset')
    parser.add_argument('--dim_ratio', default=[1.0, False],
                        help='portion of data dims selected, whether randomly selected')
    parser.add_argument('--task_seq', default=[])
    parser.add_argument('--n-task', default=0)
    parser.add_argument('--n_cls_per_task', default=2, help='how many classes does each task  contain')
    parser.add_argument('--GAT-args',
                        default={'num_layers': 1, 'num_hidden': 32, 'heads': 8, 'out_heads': 1, 'feat_drop': .6,
                                 'attn_drop': .6, 'negative_slope': 0.2, 'residual': False})
    parser.add_argument('--GCN-args', default={'h_dims': [256], 'dropout': 0.0, 'batch_norm': False})
    parser.add_argument('--GIN-args', default={'h_dims': [256], 'dropout': 0.0})
    parser.add_argument('--ergnn_args', default={'budget': 100, 'd': 0.5, 'sampler': 'CM'},
                        help='sampler options: CM, CM_plus, MF, MF_plus')
    parser.add_argument('--lwf-args', default={'lambda_dist': 1., 'T': 2})
    parser.add_argument('--twp_args', default={'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01})
    parser.add_argument('--ewc_args', default={'memory_strength': 10000.})
    parser.add_argument('--mas_args', default={'memory_strength': 10000.})
    parser.add_argument('--gem_args', default={'memory_strength': 0.5, 'n_memories': 100})
    parser.add_argument('--bare_args', default={'Na': None})
    parser.add_argument('--joint_args', default={'Na': None})
    parser.add_argument('--cls-balance', type=strtobool, default=True, help='whether to balance the cls when training and testing')
    parser.add_argument('--repeats', type=int, default=1, help='how many times to repeat the experiments for the mean and std')
    parser.add_argument('--ILmode', default='taskIL',choices=['taskIL','classIL'])
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--minibatch', type=strtobool, default=False, help='whether to use the mini-batch training')
    parser.add_argument('--batch_shuffle', type=strtobool, default=True, help='whether to shuffle the data when constructing the dataloader')
    parser.add_argument('--sample_nbs', type=strtobool, default=False, help='whether to sample neighbors instead of using all')
    parser.add_argument('--n_nbs_sample', type=lambda x: [int(i) for i in x.replace(' ', '').split(',')], default=[10, 25], help='number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25')
    parser.add_argument('--nb_sampler', default=None)
    args = parser.parse_args()
    a = [float(i) for i in args.ratio_valid_test]
    args.ratio_valid_test = a
    set_seed(args)

    if args.sample_nbs:
        args.nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.n_nbs_sample)
    else:
        args.nb_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    main = get_pipeline(args)

    method_args = {'ergnn': args.ergnn_args, 'lwf': args.lwf_args, 'twp': args.twp_args,
                   'ewc': args.ewc_args, 'bare': args.bare_args, 'gem': args.gem_args, 'mas': args.mas_args, 'joint':args.joint_args}
    backbone_args = {'GCN': args.GCN_args, 'GAT': args.GAT_args,'GIN': args.GIN_args}

    # printing AM of different sampling under different buffer ratios
    acc_matrices = []

    train_ratio = round(1-args.ratio_valid_test[0]-args.ratio_valid_test[1],2)
    if args.ILmode == 'classIL':
        if args.inter_task_edges:
            subfolder = f'inter_task_edges/cls_IL/train_ratio_{train_ratio}/'
        else:
            subfolder = f'no_inter_task_edges/cls_IL/train_ratio_{train_ratio}/'
    elif args.ILmode == 'taskIL':
        if args.inter_task_edges:
            subfolder = f'inter_task_edges/tsk_IL/train_ratio_{train_ratio}/'
        else:
            subfolder = f'no_inter_task_edges/tsk_IL/train_ratio_{train_ratio}/'

    name = f'{subfolder}{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(method_args[args.method].values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{args.repeats}'
    if args.minibatch:
        name=name+f'_bs{args.batch_size}'
    mkdir_if_missing('./results/'+subfolder)
    if os.path.isfile(
            './results/{}.pkl'.format(
                name)):
        print('the results of the following configuration exists \n',
              './results/{}.pkl'.format(
                  name))
    else:
        for ite in range(args.repeats):
            print(name, ite)
            try:
                AP, AF, acc_matrix = main(args)
                acc_matrices.append(acc_matrix)
                torch.cuda.empty_cache()
                if ite == 0:
                    with open(
                            './results/log.txt',
                            'a') as f:
                        f.write(name)
                        f.write('\nAP:{},AF:{}\n'.format(AP, AF))
            except Exception as e:
                mkdir_if_missing('./results/errors/' + subfolder)
                if ite > 0:
                    name_ = f'{subfolder}{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(method_args[args.method].values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{ite}'
                    with open(
                            './results/{}.pkl'.format(
                                name_), 'wb') as f:
                        pickle.dump(acc_matrices, f)
                print('error', e)
                name = 'errors/{}'.format(name)
                acc_matrices = traceback.format_exc()
                print(acc_matrices)
                print('error happens on \n', name)
                torch.cuda.empty_cache()
                break
        with open(
                './results/{}.pkl'.format(
                    name), 'wb') as f:
            pickle.dump(acc_matrices, f)

