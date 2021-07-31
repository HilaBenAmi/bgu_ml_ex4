"""
Incorporates mean teacher, from:

Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
Antti Tarvainen, Harri Valpola
https://arxiv.org/abs/1703.01780

"""
from bayes_opt import BayesianOptimization
import data_loaders
from sklearn.model_selection import StratifiedKFold

import time
import numpy as np
from batchup import data_source, work_pool
import network_architectures
import augmentation
import torch, torch.cuda
from torch import nn
from torch.nn import functional as F

from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import os
import pandas as pd
import datetime
import argparse

INNER_K_FOLD = 3
OUTER_K_FOLD = 10
num_epochs = 100
bo_num_iter = 50
init_points = 5
PATIENCE = 5
BEST_EPOCHS_LIST = []

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='syndigits_svhn')
args = parser.parse_args()
exp = args.exp

log_file = f'results_exp_sup_hila/log_{exp}_example_run.txt'
model_file = ''

seed = 0
device = 'cpu'
epoch_size = 'target'
batch_size = 60

torch_device = torch.device(device)
pool = work_pool.WorkerThreadPool(2)

# Setup output
def log(text):
    print(text)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(text + '\n')
            f.flush()
            f.close()


def ensure_containing_dir_exists(path):
    dir_name = os.path.dirname(path)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return path


def runner(exp):
    settings = locals().copy()

    if exp == 'svhn_mnist':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=True)
        d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False)
    elif exp == 'mnist_svhn':
        d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True)
        d_target = data_loaders.load_svhn(zero_centre=False, greyscale=True, val=False)
    elif exp == 'svhn_mnist_rgb':
        d_source = data_loaders.load_svhn(zero_centre=False, greyscale=False)
        d_target = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, val=False, rgb=True)
    elif exp == 'mnist_svhn_rgb':
        d_source = data_loaders.load_mnist(invert=False, zero_centre=False, pad32=True, rgb=True)
        d_target = data_loaders.load_svhn(zero_centre=False, greyscale=False, val=False)
    elif exp == 'cifar_stl':
        d_source = data_loaders.load_cifar10(range_01=False)
        d_target = data_loaders.load_stl(zero_centre=False, val=False)
    elif exp == 'stl_cifar':
        d_source = data_loaders.load_stl(zero_centre=False)
        d_target = data_loaders.load_cifar10(range_01=False, val=False)
    elif exp == 'mnist_usps':
        d_source = data_loaders.load_mnist(zero_centre=False)
        d_target = data_loaders.load_usps(zero_centre=False, scale28=True, val=False)
    elif exp == 'usps_mnist':
        d_source = data_loaders.load_usps(zero_centre=False, scale28=True)
        d_target = data_loaders.load_mnist(zero_centre=False, val=False)
    elif exp == 'syndigits_svhn':
        d_target = data_loaders.load_svhn(zero_centre=False, val=False)
        d_source = data_loaders.load_syn_digits(zero_centre=False)
    elif exp == 'svhn_syndigits':
        d_source = data_loaders.load_svhn(zero_centre=False, val=False)
        d_target = data_loaders.load_syn_digits(zero_centre=False)
    else:
        print('Unknown experiment type \'{}\''.format(exp))
        return

    source_x = np.array(list(d_source.train_X[:]) + list(d_source.test_X[:]))
    target_x = np.array(list(d_target.train_X[:]) + list(d_target.test_X[:]))
    source_y = np.array(list(d_source.train_y[:]) + list(d_source.test_y[:]))
    target_y = np.array(list(d_target.train_y[:]) + list(d_target.test_y[:]))
    n_classes = d_source.n_classes

    print('Loaded data')
    return source_x, source_y, target_x, target_y, n_classes


def build_and_train_model(source_train_x_inner, source_train_y_inner, target_train_x_inner, source_validation_x,
                          source_validation_y, target_validation_x, target_validation_y, learning_rate, arch='',
                          test_model=False, num_epochs=num_epochs, standardise_samples=False, affine_std=0.0,
                          xlat_range=0.0, hflip=False, intens_flip=False, intens_scale_range='', intens_offset_range='',
                          gaussian_noise_std=0.0):


    net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)

    settings = locals().copy()

    intens_scale_range_lower, intens_scale_range_upper, intens_offset_range_lower, intens_offset_range_upper = \
        None, None, None, None


    if expected_shape != source_train_x_inner.shape[1:]:
        print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
              'data has samples of shape {}'.format(arch, exp, expected_shape, source_train_x_inner.shape[1:]))
        return

    net = net_class(n_classes).to(torch_device)
    params = list(net.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    classification_criterion = nn.CrossEntropyLoss()

    print('Built network')

    aug = augmentation.ImageAugmentation(
        hflip, xlat_range, affine_std,
        intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
        intens_offset_range_lower=intens_offset_range_lower, intens_offset_range_upper=intens_offset_range_upper,
        intens_flip=intens_flip, gaussian_noise_std=gaussian_noise_std)

    def augment(X_sup, y_sup):
        X_sup = aug.augment(X_sup)
        return [X_sup, y_sup]

    def f_train(X_sup, y_sup):
        X_sup = torch.autograd.Variable(torch.from_numpy(X_sup).float().to(torch_device))
        y_sup = torch.autograd.Variable(torch.from_numpy(y_sup).long().to(torch_device))

        optimizer.zero_grad()
        net.train(mode=True)

        sup_logits_out = net(X_sup)

        # Supervised classification loss
        clf_loss = classification_criterion(sup_logits_out, y_sup)

        loss_expr = clf_loss

        loss_expr.backward()
        optimizer.step()

        n_samples = X_sup.size()[0]

        return float(clf_loss.data.cpu().numpy()) * n_samples

    print('Compiled training function')

    def f_pred(X_sup):
        X_var = torch.autograd.Variable(torch.from_numpy(X_sup).float().to(torch_device)
                                        )
        net.train(mode=False)
        return F.softmax(net(X_var)).data.cpu().numpy()

    def f_eval(X_sup, y_sup):
        y_pred_prob = f_pred(X_sup)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return float((y_pred != y_sup).sum())

    def f_pred_for_metrics(X_sup, y_sup):
        y_pred_prob = f_pred(X_sup)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return (y_pred, y_pred_prob)

    print('Compiled evaluation function')

    ensure_containing_dir_exists(log_file)

    # Report setttings
    log(f'learning_rate={learning_rate}')


    print('Training...')
    train_ds = data_source.ArrayDataSource([source_train_x_inner, source_train_y_inner]).map(augment)

    source_test_ds = data_source.ArrayDataSource([source_validation_x, source_validation_y])
    target_test_ds = data_source.ArrayDataSource([target_validation_x, target_validation_y])

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    best_src_test_err = 1.0
    best_epoch = 0
    count_no_improve = 0
    count_no_improve_flag = False
    t_training_1 = time.time()
    for epoch in range(num_epochs):
        t1 = time.time()

        train_res = train_ds.batch_map_mean(f_train, batch_size=batch_size, shuffle=shuffle_rng)

        train_clf_loss = train_res[0]
        src_test_err, = source_test_ds.batch_map_mean(f_eval, batch_size=batch_size * 4)
        tgt_test_err, = target_test_ds.batch_map_mean(f_eval, batch_size=batch_size * 4)

        if src_test_err < best_src_test_err:
            improve = '*** '
            best_epoch = epoch
            best_src_test_err = src_test_err
            if count_no_improve_flag:
                count_no_improve_flag = False
                count_no_improve = 0
        else:
            improve = ''
            count_no_improve_flag = True
            count_no_improve += 1

        t2 = time.time()

        log('{}Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
            'SRC TEST ERR={:.3%}, TGT TEST err={:.3%}'.format(
            improve, epoch, t2 - t1, train_clf_loss, src_test_err, tgt_test_err))
        if count_no_improve >= PATIENCE:
            break
    t_training_2 = time.time()
    if test_model:
        t_inference_1 = time.time()
        src_pred, src_prob = source_test_ds.batch_map_concat(f_pred_for_metrics, batch_size=batch_size * 4)
        t_inference_2 = time.time()
        tgt_pred, tgt_prob = target_test_ds.batch_map_concat(f_pred_for_metrics, batch_size=batch_size * 4)
        src_scores_dict = create_metrics_results(source_validation_y, src_pred, src_prob)
        tgt_scores_dict = create_metrics_results(target_validation_y, tgt_pred, tgt_prob)
        inference_time_for_1000 = (t_inference_2-t_inference_1)/len(src_pred)*1000
        return src_scores_dict, tgt_scores_dict, round(t_training_2-t_training_1, 3), round(inference_time_for_1000, 4)
    return best_src_test_err, best_epoch


def calc_metrics(sup_y, sup_y_one_hot, pred_y, prob, class_labels):
    scores_dict = {}
    conf = confusion_matrix(sup_y, pred_y)
    tpr_list = []
    fpr_list = []
    for label in range(conf.shape[0]):
        tpr = conf[label][label] / sum(conf[label])
        tpr_list.append(tpr)
        fpr_numerator = sum([pred_row[label] for pred_row in conf]) - conf[label][label]
        fpr_denominator = sum(sum(conf)) - sum(conf[label])
        fpr_list.append(fpr_numerator / fpr_denominator)
    scores_dict['tpr'] = np.round(np.mean(tpr_list), 4)
    scores_dict['fpr'] = np.round(np.mean(fpr_list), 4)

    # classification_report = metrics.classification_report(sup_y, pred_y, digits=3)
    # print(classification_report)
    scores_dict['acc'] = np.round(metrics.accuracy_score(sup_y, pred_y), 2)
    scores_dict['roc_auc'] = np.round(metrics.roc_auc_score(sup_y, prob, multi_class='ovr'), 4)
    scores_dict['precision'] = np.round(metrics.precision_score(sup_y, pred_y, average='macro'), 4)

    pred_one_hot = label_binarize(pred_y, classes=class_labels)
    scores_dict['recall_precision_auc'] = np.round(metrics.average_precision_score(sup_y_one_hot, pred_one_hot, average="macro"), 4)
    scores_dict['err'] = float((pred_y != sup_y).mean())
    return scores_dict


def create_metrics_results(sup_y, pred, prob):
    class_labels = list(np.unique(sup_y))
    sup_y_one_hot = label_binarize(sup_y, classes=class_labels)
    scores_dict = calc_metrics(sup_y, sup_y_one_hot, pred, prob, class_labels)
    return scores_dict


def get_arch(exp, arch):
    if arch == '':
        if exp in {'mnist_usps', 'usps_mnist'}:
            arch = 'mnist-bn-32-64-256'
        elif exp in {'svhn_mnist', 'mnist_svhn'}:
            arch = 'mnist-bn-32-32-64-256'
        if exp in {'cifar_stl', 'stl_cifar', 'syndigits_svhn', 'svhn_syndigits', 'svhn_mnist_rgb', 'mnist_svhn_rgb'}:
            arch = 'mnist-bn-32-32-64-256-rgb'
    return arch


def evaluate_exp(learning_rate, arch=''):

    arch = get_arch(exp, arch)

    cv_source = StratifiedKFold(n_splits=INNER_K_FOLD, shuffle=True)
    source_train_validation_list = []
    for train_idx, validation_idx in cv_source.split(source_train_x, source_train_y):
        source_dict = {}
        train_data, validation_data = source_train_x[train_idx], source_train_x[validation_idx]
        train_target, validation_target = source_train_y[train_idx], source_train_y[validation_idx]
        source_dict['source_train_x'] = train_data
        source_dict['source_train_y'] = train_target
        source_dict['source_validation_x'] = validation_data
        source_dict['source_validation_y'] = validation_target
        source_train_validation_list.append(source_dict)

    cv_target = StratifiedKFold(n_splits=INNER_K_FOLD, shuffle=True)
    target_train_validation_list = []
    for train_idx, validation_idx in cv_target.split(target_train_x, target_train_y):
        target_dict = {}
        train_data, validation_data = target_train_x[train_idx], target_train_x[validation_idx]
        validation_target = target_train_y[validation_idx]
        target_dict['target_train_x'] = train_data
        target_dict['target_validation_x'] = validation_data
        target_dict['target_validation_y'] = validation_target
        target_train_validation_list.append(target_dict)

    best_src_test_err_list = []
    best_epoch_list = []
    for cv_idx in range(INNER_K_FOLD):
        print(f'start inner cv {cv_idx}')
        log(f'start inner cv {cv_idx}')
        source_train_x_inner = source_train_validation_list[cv_idx]['source_train_x']
        source_train_y_inner = source_train_validation_list[cv_idx]['source_train_y']
        source_validation_x = source_train_validation_list[cv_idx]['source_validation_x']
        source_validation_y = source_train_validation_list[cv_idx]['source_validation_y']
        target_train_x_inner = target_train_validation_list[cv_idx]['target_train_x']
        target_validation_x = target_train_validation_list[cv_idx]['target_validation_x']
        target_validation_y = target_train_validation_list[cv_idx]['target_validation_y']

        if cv_idx == 0:
            # Report dataset size
            log('Dataset:')
            log('SOURCE Train: X.shape={}, y.shape={}'.format(source_train_x_inner.shape, source_train_y_inner.shape))
            log('SOURCE Val: X.shape={}, y.shape={}'.format(source_validation_x.shape, source_validation_y.shape))
            log('TARGET Train: X.shape={}'.format(target_train_x_inner.shape))
            log('TARGET Val: X.shape={}, y.shape={}'.format(target_validation_x.shape, target_validation_y.shape))

        best_src_test_err, best_epoch = build_and_train_model(
            source_train_x_inner, source_train_y_inner, target_train_x_inner,
            source_validation_x, source_validation_y, target_validation_x, target_validation_y,
            learning_rate, arch)
        best_src_test_err_list.append(best_src_test_err)
        best_epoch_list.append(best_epoch)

    BEST_EPOCHS_LIST.append(int(np.mean(best_epoch_list)))
    return -np.mean(best_src_test_err_list)


def rebuild_and_test_model(params, source_train_x, source_train_y, target_train_x, source_test_x,
                           source_test_y, target_test_x, target_test_y, arch=''):
    log(f"Start rebuild on test set")
    arch = get_arch(exp, arch)
    best_epoch = np.max(BEST_EPOCHS_LIST)
    log(f'best_epoch: {best_epoch}')
    results = build_and_train_model(
        source_train_x, source_train_y, target_train_x, source_test_x, source_test_y, target_test_x, target_test_y,
        arch=arch, test_model=True, num_epochs=best_epoch, **params)
    src_scores_dict = results[0]
    tgt_scores_dict = results[1]
    training_time = results[2]
    inference_time = results[3]
    log(f'src_scores_dict: {src_scores_dict}')
    log(f'tgt_scores_dict: {tgt_scores_dict}')
    log(f'training_time: {training_time}')
    log(f'inference_time for 1000 instences: {inference_time}')
    src_scores = pd.Series(src_scores_dict)
    tgt_scores_dict.update({'training_time': training_time, 'inference_time': inference_time, 'params': params})
    tgt_scores = pd.Series(tgt_scores_dict)
    return src_scores, tgt_scores


if __name__ == '__main__':
    global source_train_x, source_train_y, target_train_x, target_train_y

    source_x, source_y, target_x, target_y, n_classes = runner(exp=exp)

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None
    ensure_containing_dir_exists(log_file)

    cv_source = StratifiedKFold(n_splits=OUTER_K_FOLD, shuffle=True)
    source_train_test_list = []
    for train_idx, test_idx in cv_source.split(source_x, source_y):
        source_dict = {}
        train_data, test_data = source_x[train_idx], source_x[test_idx]
        train_target, test_target = source_y[train_idx], source_y[test_idx]
        source_dict['source_train_x'] = train_data[:1000]
        source_dict['source_train_y'] = train_target[:1000]
        source_dict['source_test_x'] = test_data[:300]
        source_dict['source_test_y'] = test_target[:300]
        source_train_test_list.append(source_dict)

    cv_target = StratifiedKFold(n_splits=OUTER_K_FOLD, shuffle=True)
    target_train_test_list = []
    for train_idx, test_idx in cv_target.split(target_x, target_y):
        target_dict = {}
        train_data, test_data = target_x[train_idx], target_x[test_idx]
        train_target, test_target = target_y[train_idx], target_y[test_idx]
        target_dict['target_train_x'] = train_data[:1000]
        target_dict['target_train_y'] = train_target[:1000]
        target_dict['target_test_x'] = test_data[:300]
        target_dict['target_test_y'] = test_target[:300]
        target_train_test_list.append(target_dict)

    src_scores_list = []
    tgt_scores_list = []
    try:
        for cv_idx in range(OUTER_K_FOLD):
            print(f'start outer cv {cv_idx}')
            log(f'start outer cv {cv_idx}')
            source_train_x = source_train_test_list[cv_idx]['source_train_x']
            source_train_y = source_train_test_list[cv_idx]['source_train_y']
            source_test_x = source_train_test_list[cv_idx]['source_test_x']
            source_test_y = source_train_test_list[cv_idx]['source_test_y']
            target_train_x = target_train_test_list[cv_idx]['target_train_x']
            target_train_y = target_train_test_list[cv_idx]['target_train_y']
            target_test_x = target_train_test_list[cv_idx]['target_test_x']
            target_test_y = target_train_test_list[cv_idx]['target_test_y']

            if cv_idx == 0:
                # Report dataset size
                log('Dataset:')
                log('SOURCE Train: X.shape={}, y.shape={}'.format(source_train_x.shape, source_train_y.shape))
                log('SOURCE Test: X.shape={}, y.shape={}'.format(source_test_x.shape, source_test_y.shape))
                log('TARGET Train: X.shape={}'.format(target_train_x.shape))
                log('TARGET Test: X.shape={}, y.shape={}'.format(target_test_x.shape, target_test_y.shape))

            domain_adapt_BO = BayesianOptimization(evaluate_exp, {'learning_rate': (0.00001, 0.1)})
            domain_adapt_BO.maximize(init_points=init_points, n_iter=bo_num_iter)
            params_domain_adapt = domain_adapt_BO.max['params']

            log(f'outer CV {cv_idx} - Opt hyper params: {params_domain_adapt} \n with target of: {domain_adapt_BO.max["target"]}')

            src_scores, tgt_scores = rebuild_and_test_model(params_domain_adapt, source_train_x, source_train_y, target_train_x, source_test_x,
                               source_test_y, target_test_x, target_test_y)
            src_scores_list.append(src_scores)
            tgt_scores_list.append(tgt_scores)
            log('*********************************************************************** \n')
    except Exception:
        log(f"stopped at cv: {cv_idx}")
        raise
    finally:
        if len(tgt_scores_list) > 0:
            tgt_scores_df = pd.DataFrame(tgt_scores_list)
            src_scores_df = pd.DataFrame(src_scores_list)
            date = datetime.datetime.now().strftime('%d-%m_%H-%M')
            tgt_scores_df.to_csv(f"./sup/tgt_scores_df_{exp}_until_cv_{cv_idx}_{date}.csv")
            src_scores_df.to_csv(f"./sup/src_scores_df_{exp}_until_cv_{cv_idx}_{date}.csv")