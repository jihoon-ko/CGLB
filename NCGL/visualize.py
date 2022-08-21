import pickle
import matplotlib.pyplot as plt
import numpy as np

def AM_err(acc_matrices):
    # given a list of acc matrices, return the AMs and the errors
    n_tasks = acc_matrices[0].shape[0]
    acc_means_all_repeats = np.stack([[np.mean(m[i,0:i+1]) for i in range(n_tasks)] for m in acc_matrices])
    acc_means = np.mean(acc_means_all_repeats, axis=0)
    err_all = acc_means_all_repeats - acc_means
    std = acc_means_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return acc_means, err, std

def FM(acc_matrix):
    # given a acc matrix, return FM
    n_tasks = acc_matrix.shape[0]
    backward = []
    for t in range(n_tasks - 1):
        b = acc_matrix[n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(b)
    return np.mean(backward)

def FM_err(acc_matrices):
    # given a list of acc matrices, return the AMs and the errors
    FM_all_repeats = []
    for m in acc_matrices:
        FM_all_repeats.append(FM(m))
    FM_mean = np.mean(FM_all_repeats)
    FM_all_repeats = np.stack(FM_all_repeats)
    err_all = FM_all_repeats - FM_mean
    std = FM_all_repeats.std(0)
    err_plus = err_all.max(axis=0)
    err_minus = err_all.min(axis=0).__abs__()
    err = np.stack([err_minus, err_plus])
    return FM_mean, err, std

def show_performance_matrices(result_path, save_fig_name=None):
    """
    The function to visualize the performance matrix.

    :param result_path: The path to the experimental result
    :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
    """
    # visualize the acc matrices
    print(result_path)
    fig, ax = plt.subplots()
    acc_matrices = pickle.load(open(result_path, 'rb'))
    acc_matrix_mean = np.mean(acc_matrices, axis=0)
    mask = np.tri(acc_matrix_mean.shape[0], k=-1).T
    acc_matrix_mean = np.ma.array(acc_matrix_mean, mask=mask)
    im = plt.imshow(acc_matrix_mean)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel('$\mathrm{Tasks}$')
    plt.ylabel('$\mathrm{Tasks}$')
    plt.clim(vmin=0, vmax=100)
    cbar = fig.colorbar(im, ticks=[0, 50, 100])  # , fontsize = 15)
    cbar.ax.tick_params()

    if save_fig_name is not None:
        plt.savefig(f'./results/figures/{save_fig_name}_performance_matrix', bbox_inches='tight')
    plt.show()

def show_learning_curve(result_path, save_fig_name=None):
    """
        The function to visualize the dynamics of AP.

        :param result_path: The path to the experimental result
        :param save_fig_name: If specified, the generated visualization will be stored with the specified name under the directory "./results/figures"
        """
    #to draw AP against buffer task with different methods
    print(result_path)
    acc_matrices = pickle.load(open(result_path, 'rb'))
    acc_mean, err, _ = AM_err(acc_matrices)
    x = list(range(len(acc_mean)))
    plt.errorbar(x, acc_mean)
    if save_fig_name is not None:
        plt.savefig(
            f'./results/figures/{save_fig_name}_learning_curve', bbox_inches='tight')
    plt.show()

def show_final_APAF(result_path):
    """
        The function to show the final AP and AF. Output are orgnized in a LaTex firendly way.

        :param result_path: The path to the experimental result
        """
    #show the final AP and AF
    acc_matrices = pickle.load(open(result_path, 'rb'))
    acc_mean, err, std_am = AM_err(acc_matrices)
    # err_minus,err_plus = err[0,:], err[1, :]

    # AF
    # FM_means = [FM_err(m)[0] for m in acc_matrices]
    FM_mean, err_FM, std_fm = FM_err(acc_matrices)
    # err_FM = np.stack([FM_err(m)[1] for m in acc_matrices]).transpose(1, 0)
    print(r'{:.1f}$\pm${:.1f}&{:.1f}$\pm${:.1f}'.format(acc_mean[-1], std_am[-1], FM_mean, std_fm))


