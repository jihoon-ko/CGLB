import traceback
import pickle
from distutils.util import strtobool
from pipeline import *

if __name__ == '__main__':
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='G-CGL')
    parser.add_argument('--backbone', type=str, default='GCN', choices=['CusGCN','GCN', 'GAT', 'Weave', 'HPNs'],
                        help='Model to use')
    parser.add_argument('--method', type=str, choices=['bare', 'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain', 'jointreplay'],
                        default='bare', help='Method to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['SIDER-tIL','Tox21-tIL','Aromaticity-CL'], default='SIDER-tIL',
                        help='Dataset to use')
    parser.add_argument('-p', '--pre-trained', action='store_true',
                        help='Whether to skip training and use a pre-trained model')
    parser.add_argument('-g', '--gpu', type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")

    # ewc/mas/gem
    parser.add_argument('-me', '--memory-strength', type=float, default=10000,
                        help="memory strength, 10000 for ewc/mas, 0.5 for gem")
    # gem
    parser.add_argument('-n', '--n-memories', type=int, default=100,
                        help="number of memories")

    # parameters for our method (twp)
    parser.add_argument('-l', '--lambda_l', type=float, default=10000,
                        help=" ")    
    parser.add_argument('-t', '--lambda_t', type=float, default=100,
                        help=" ")    
    parser.add_argument('-b', '--beta', type=float, default=0.1,
                        help=" ")

    parser.add_argument('-s', '--random_seed', type=int, default=0,
                        help="seed for exp")
    parser.add_argument('--alpha_dis', type=float, default=0.1)
    parser.add_argument('--classifier_increase',default=False)
    parser.add_argument('--clsIL', type=strtobool, default=False)
    parser.add_argument('--n_cls_per_task', default=1)
    parser.add_argument('--num_epochs',type=int,default=2)
    parser.add_argument('--threshold_pubchem', default=20)
    parser.add_argument('--frac_train', default=0.8)
    parser.add_argument('--frac_val', default=0.1)
    parser.add_argument('--frac_test', default=0.1)
    parser.add_argument('--repeats', default=1)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=True)

    args = parser.parse_args().__dict__
    args['exp'] = 'config'
    args['gem_args'] = [args['memory_strength'],args['n_memories']]
    args['ewc_args']= [args['memory_strength']]
    args['mas_args']= [args['memory_strength']]
    args['twp_args'] = [args['lambda_l'],args['lambda_t'],args['beta']]
    args.update(get_exp_configure(args['exp']))

    method_args = {'lwf': [], 'twp': args['twp_args'],'jointtrain':[],'jointreplay':[],
                   'ewc': args['ewc_args'], 'bare': [], 'gem': args['gem_args'], 'mas': args['mas_args']}
    acc_matrices = []
    main = get_pipeline(args)
    if args['dataset'] in ['Aromaticity-CL']:
        args['n_cls_per_task'] = 2
        if args['clsIL']:
            if args['method'] == 'jointtrain':
                args['method'] = 'jointreplay'
            subfolder = f"clsIL/{args['frac_train']}/"
        else:
            subfolder = f'tskIL/{args["frac_train"]}/'
    else:
        args['n_cls_per_task'] =1
        subfolder = f'tskIL/{args["frac_train"]}/'

    name = '{}{}_{}_{}_{}_{}_{}_{}_{}'.format(subfolder, args['dataset'], args['n_cls_per_task'], args['method'],
                                              method_args[args['method']],
                                              args['backbone'],
                                              args['gcn_hidden_feats'],
                                              args['classifier_increase'], args['repeats'])
    mkdir_if_missing('./results/' + subfolder)
    if args['replace_illegal_char']:
        name = remove_illegal_characters(name)
    if os.path.isfile('./results/{}.pkl'.format(name)):
        print('the following configuration exists \n','./results/{}.pkl'.format(name))
    else:
        for ite in range(args['repeats']):
            print(name, ite)
            try:
                AP, AF, acc_matrix = main(args)
                acc_matrices.append(acc_matrix)
                torch.cuda.empty_cache()
                if ite == 0:
                    with open(
                            './results/log.txt', 'a') as f:
                        f.write(name)
                        f.write('\nAP:{},AF:{}\n'.format(AP, AF))
            except Exception as e:
                mkdir_if_missing('./results/errors/' + subfolder)
                if ite > 0:
                    name_ = '{}{}_{}_{}_{}_{}_{}_{}_{}'.format(subfolder, args['dataset'],
                                                               args['n_cls_per_task'],
                                                               args['method'], method_args[args['method']],
                                                               args['backbone'], args['gcn_hidden_feats'],
                                                               args['classifier_increase'],
                                                               ite)
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


