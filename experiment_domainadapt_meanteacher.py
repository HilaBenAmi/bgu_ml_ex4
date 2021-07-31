"""
Code was adapted from https://github.com/Britefury/self-ensemble-visual-domain-adapt
"""


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
import optim_weight_ema

from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import os
import pickle
import datetime
import pandas as pd
import cmdline_helpers
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

log_file = f'results_exp_meanteacher_hila/log_{exp}_run.txt'
model_file = ''

seed = 0
device = 'cpu'
epoch_size = 'target'
batch_size = 30


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


def load_data(exp):
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
        d_source = data_loaders.load_syn_digits(zero_centre=False)
        d_target = data_loaders.load_svhn(zero_centre=False, val=False)
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


def build_and_train_model(source_train_x_inner, source_train_y_inner, target_train_x_inner,
                source_validation_x, source_validation_y, target_validation_x, target_validation_y,
               confidence_thresh, teacher_alpha, unsup_weight, cls_balance, learning_rate, arch='', test_model=False,
               loss='var', num_epochs=num_epochs, cls_bal_scale=False, cls_bal_scale_range=0.0, cls_balance_loss='bce',
               src_affine_std=0.0, src_xlat_range=0.0, src_hflip=False, src_intens_flip=False,
               src_gaussian_noise_std=0.0, tgt_affine_std=0.0, tgt_xlat_range=0.0, tgt_hflip=False,
               tgt_intens_flip='', tgt_gaussian_noise_std=0.0):

    net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)

    settings = locals().copy()

    src_intens_scale_range_lower, src_intens_scale_range_upper, src_intens_offset_range_lower, src_intens_offset_range_upper = \
        None, None, None, None
    tgt_intens_scale_range_lower, tgt_intens_scale_range_upper, tgt_intens_offset_range_lower, tgt_intens_offset_range_upper = \
        None, None, None, None

    if expected_shape != source_train_x_inner.shape[1:]:
        print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
              'data has samples of shape {}'.format(arch, exp, expected_shape, source_train_x_inner.shape[1:]))
        return

    student_net = net_class(n_classes).to(torch_device)
    teacher_net = net_class(n_classes).to(torch_device)
    student_params = list(student_net.parameters())
    teacher_params = list(teacher_net.parameters())
    for param in teacher_params:
        param.requires_grad = False

    student_optimizer = torch.optim.Adam(student_params, lr=learning_rate)
    teacher_optimizer = optim_weight_ema.OldWeightEMA(teacher_net, student_net, alpha=teacher_alpha)
    classification_criterion = nn.CrossEntropyLoss()

    print('Built network')

    src_aug = augmentation.ImageAugmentation(
        src_hflip, src_xlat_range, src_affine_std,
        intens_flip=src_intens_flip,
        intens_scale_range_lower=src_intens_scale_range_lower, intens_scale_range_upper=src_intens_scale_range_upper,
        intens_offset_range_lower=src_intens_offset_range_lower,
        intens_offset_range_upper=src_intens_offset_range_upper,
        gaussian_noise_std=src_gaussian_noise_std
    )
    tgt_aug = augmentation.ImageAugmentation(
        tgt_hflip, tgt_xlat_range, tgt_affine_std,
        intens_flip=tgt_intens_flip,
        intens_scale_range_lower=tgt_intens_scale_range_lower, intens_scale_range_upper=tgt_intens_scale_range_upper,
        intens_offset_range_lower=tgt_intens_offset_range_lower,
        intens_offset_range_upper=tgt_intens_offset_range_upper,
        gaussian_noise_std=tgt_gaussian_noise_std
    )

    def augment(X_src, y_src, X_tgt):
        X_src = src_aug.augment(X_src)
        X_tgt_stu, X_tgt_tea = tgt_aug.augment_pair(X_tgt)
        return X_src, y_src, X_tgt_stu, X_tgt_tea

    rampup_weight_in_list = [0]

    cls_bal_fn = network_architectures.get_cls_bal_function(cls_balance_loss)

    def compute_aug_loss(stu_out, tea_out):
        # Augmentation loss
        conf_tea = torch.max(tea_out, 1)[0]
        unsup_mask = conf_mask = (conf_tea > confidence_thresh).float()
        unsup_mask_count = conf_mask_count = conf_mask.sum()

        if loss == 'bce':
            aug_loss = network_architectures.robust_binary_crossentropy(stu_out, tea_out)
        else:
            d_aug_loss = stu_out - tea_out
            aug_loss = d_aug_loss * d_aug_loss

        # Class balance scaling
        if cls_bal_scale:
            n_samples = unsup_mask.sum()
            avg_pred = n_samples / float(n_classes)
            bal_scale = avg_pred / torch.clamp(tea_out.sum(dim=0), min=1.0)
            if cls_bal_scale_range != 0.0:
                bal_scale = torch.clamp(bal_scale, min=1.0 / cls_bal_scale_range, max=cls_bal_scale_range)
            bal_scale = bal_scale.detach()
            aug_loss = aug_loss * bal_scale[None, :]

        aug_loss = aug_loss.mean(dim=1)
        unsup_loss = (aug_loss * unsup_mask).mean()

        # Class balance loss
        if cls_balance > 0.0:
            # Compute per-sample average predicated probability
            # Average over samples to get average class prediction
            avg_cls_prob = stu_out.mean(dim=0)
            # Compute loss
            equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))
            equalise_cls_loss = equalise_cls_loss.mean() * n_classes
            equalise_cls_loss = equalise_cls_loss * unsup_mask.mean(dim=0)
            unsup_loss += equalise_cls_loss * cls_balance

        return unsup_loss, conf_mask_count, unsup_mask_count

    def f_train(X_src, y_src, X_tgt0, X_tgt1):
        X_src = torch.tensor(X_src, dtype=torch.float, device=torch_device)
        y_src = torch.tensor(y_src, dtype=torch.long, device=torch_device)
        X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float, device=torch_device)
        X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float, device=torch_device)

        student_optimizer.zero_grad()
        student_net.train()
        teacher_net.train()

        src_logits_out = student_net(X_src)
        student_tgt_logits_out = student_net(X_tgt0)
        student_tgt_prob_out = F.softmax(student_tgt_logits_out, dim=1)
        teacher_tgt_logits_out = teacher_net(X_tgt1)
        teacher_tgt_prob_out = F.softmax(teacher_tgt_logits_out, dim=1)

        # Supervised classification loss
        clf_loss = classification_criterion(src_logits_out, y_src)

        unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_tgt_prob_out, teacher_tgt_prob_out)

        loss_expr = clf_loss + unsup_loss * unsup_weight

        loss_expr.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        n_samples = X_src.size()[0]

        outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_samples]
        return tuple(outputs)

    print('Compiled training function')

    def f_pred(X_sup):
        X_var = torch.tensor(X_sup, dtype=torch.float, device=torch_device)
        student_net.eval()
        teacher_net.eval()
        return (F.softmax(student_net(X_var), dim=1).detach().cpu().numpy(),
                F.softmax(teacher_net(X_var), dim=1).detach().cpu().numpy())

    def f_eval(X_sup, y_sup):
        y_pred_prob_stu, y_pred_prob_tea = f_pred(X_sup)
        y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
        y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
        return (float((y_pred_stu != y_sup).sum()), float((y_pred_tea != y_sup).sum()))

    def f_pred_for_metrics(X_sup, y_sup):
        y_pred_prob_stu, y_pred_prob_tea = f_pred(X_sup)
        y_pred_stu = np.argmax(y_pred_prob_stu, axis=1)
        y_pred_tea = np.argmax(y_pred_prob_tea, axis=1)
        return (y_pred_stu, y_pred_tea, y_pred_prob_stu, y_pred_prob_tea)

    print('Compiled evaluation function')

    cmdline_helpers.ensure_containing_dir_exists(log_file)

    # Report setttings
    log(f'confidence_thresh={confidence_thresh}, teacher_alpha={teacher_alpha},\
     unsup_weight={unsup_weight}, cls_balance={cls_balance}, learning_rate={learning_rate}, num_epochs={num_epochs}'
        f'test_model={test_model}')

    print('Training...')
    sup_ds = data_source.ArrayDataSource([source_train_x_inner, source_train_y_inner], repeats=-1)
    tgt_train_ds = data_source.ArrayDataSource([target_train_x_inner], repeats=-1)
    train_ds = data_source.CompositeDataSource([sup_ds, tgt_train_ds]).map(augment)
    train_ds = pool.parallel_data_source(train_ds)
    if epoch_size == 'large':
        n_samples = max(source_train_x_inner.shape[0], target_train_x_inner.shape[0])
    elif epoch_size == 'small':
        n_samples = min(source_train_x_inner.shape[0], target_train_x_inner.shape[0])
    elif epoch_size == 'target':
        n_samples = target_train_x_inner.shape[0]
    n_train_batches = n_samples // batch_size

    source_test_ds = data_source.ArrayDataSource([source_validation_x, source_validation_y])
    target_test_ds = data_source.ArrayDataSource([target_validation_x, target_validation_y])

    if seed != 0:
        shuffle_rng = np.random.RandomState(seed)
    else:
        shuffle_rng = np.random

    train_batch_iter = train_ds.batch_iterator(batch_size=batch_size, shuffle=shuffle_rng)

    best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}

    best_conf_mask_rate = 0.0
    best_src_test_err = 1.0
    best_target_tea_err = 1.0
    best_epoch = 0
    count_no_improve = 0
    count_no_improve_flag = False
    t_training_1 = time.time()
    for epoch in range(num_epochs):
        t1 = time.time()
        train_res = data_source.batch_map_mean(f_train, train_batch_iter, n_batches=n_train_batches)
        train_clf_loss = train_res[0]
        unsup_loss_string = 'unsup (tgt) loss={:.6f}'.format(train_res[1])

        src_test_err_stu, src_test_err_tea = source_test_ds.batch_map_mean(f_eval, batch_size=batch_size * 2)
        tgt_test_err_stu, tgt_test_err_tea = target_test_ds.batch_map_mean(f_eval, batch_size=batch_size * 2)

        conf_mask_rate = train_res[-2]
        unsup_mask_rate = train_res[-1]
        if conf_mask_rate > best_conf_mask_rate:
            best_conf_mask_rate = conf_mask_rate
            improve = '*** '
            best_teacher_model_state = {k: v.cpu().numpy() for k, v in teacher_net.state_dict().items()}
            best_target_tea_err = tgt_test_err_tea
            best_epoch = epoch
            if count_no_improve_flag:
                count_no_improve_flag = False
                count_no_improve = 0
        else:
            improve = ''
            count_no_improve_flag = True
            count_no_improve += 1
        unsup_loss_string = '{}, conf mask={:.3%}, unsup mask={:.3%}'.format(
            unsup_loss_string, conf_mask_rate, unsup_mask_rate)
        t2 = time.time()

        log('{}Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}, {}; '
            'SRC TEST ERR={:.3%}, TGT TEST student err={:.3%}, TGT TEST teacher err={:.3%}'.format(
            improve, epoch, t2 - t1, train_clf_loss, unsup_loss_string, src_test_err_stu, tgt_test_err_stu,
            tgt_test_err_tea))
        if count_no_improve >= PATIENCE:
            break
    t_training_2 = time.time()
    if test_model:
        t_inference_1 = time.time()
        src_pred_stu, src_pred_tea, src_prob_stu, src_prob_tea = source_test_ds.batch_map_concat(f_pred_for_metrics, batch_size=batch_size * 2)
        t_inference_2 = time.time()
        tgt_pred_stu, tgt_pred_tea, tgt_prob_stu, tgt_prob_tea = target_test_ds.batch_map_concat(f_pred_for_metrics, batch_size=batch_size * 2)
        src_stu_scores_dict, src_tea_scores_dict = create_metrics_results(source_validation_y, src_pred_stu, src_pred_tea, src_prob_stu, src_prob_tea)
        tgt_stu_scores_dict, tgt_tea_scores_dict = create_metrics_results(target_validation_y, tgt_pred_stu, tgt_pred_tea, tgt_prob_stu, tgt_prob_tea)
        inference_time_for_1000 = (t_inference_2-t_inference_1)/len(src_pred_stu)*1000
        return src_stu_scores_dict, src_tea_scores_dict, tgt_stu_scores_dict, tgt_tea_scores_dict, round(t_training_2-t_training_1, 3), round(inference_time_for_1000, 4)
    return best_target_tea_err, best_teacher_model_state, best_epoch


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

    scores_dict['acc'] = np.round(metrics.accuracy_score(sup_y, pred_y), 2)
    scores_dict['roc_auc'] = np.round(metrics.roc_auc_score(sup_y, prob, multi_class='ovr'), 4)
    scores_dict['precision'] = np.round(metrics.precision_score(sup_y, pred_y, average='macro'), 4)

    pred_one_hot = label_binarize(pred_y, classes=class_labels)
    scores_dict['recall_precision_auc'] = np.round(metrics.average_precision_score(sup_y_one_hot, pred_one_hot, average="macro"), 4)
    scores_dict['err'] = float((pred_y != sup_y).mean())
    return scores_dict


def create_metrics_results(sup_y, pred_stu, pred_tea, prob_stu, prob_tea):
    class_labels = list(np.unique(sup_y))
    sup_y_one_hot = label_binarize(sup_y, classes=class_labels)
    stu_scores_dict = calc_metrics(sup_y, sup_y_one_hot, pred_stu, prob_stu, class_labels)
    tea_scores_dict = calc_metrics(sup_y, sup_y_one_hot, pred_tea, prob_tea, class_labels)
    return stu_scores_dict, tea_scores_dict


def get_arch(exp, arch):
    if arch == '':
        if exp in {'mnist_usps', 'usps_mnist'}:
            arch = 'mnist-bn-32-64-256'
        elif exp in {'svhn_mnist', 'mnist_svhn'}:
            arch = 'mnist-bn-32-32-64-256'
        if exp in {'cifar_stl', 'stl_cifar', 'syndigits_svhn', 'svhn_syndigits', 'svhn_mnist_rgb', 'mnist_svhn_rgb'}:
            arch = 'mnist-bn-32-32-64-256-rgb'
    return arch


def evaluate_exp(confidence_thresh, teacher_alpha, unsup_weight, cls_balance, learning_rate, arch=''):
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

    target_test_err_list = []
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

        best_target_tea_err, best_teacher_model_state, best_epoch = build_and_train_model(
            source_train_x_inner, source_train_y_inner, target_train_x_inner,
            source_validation_x, source_validation_y, target_validation_x, target_validation_y,
            confidence_thresh, teacher_alpha, unsup_weight, cls_balance, learning_rate,
            arch)
        target_test_err_list.append(best_target_tea_err)
        best_epoch_list.append(best_epoch)

        # Save network
        if model_file != '':
            cmdline_helpers.ensure_containing_dir_exists(model_file)
            with open(model_file, 'wb') as f:
                pickle.dump(best_teacher_model_state, f)

    BEST_EPOCHS_LIST.append(int(np.mean(best_epoch_list)))
    return -np.mean(target_test_err_list)


def rebuild_and_test_model(params, source_train_x, source_train_y, target_train_x, source_test_x,
                           source_test_y, target_test_x, target_test_y, arch=''):
    log(f"Start rebuild on test set")
    arch = get_arch(exp, arch)
    best_epoch = np.max(BEST_EPOCHS_LIST)
    log(f'best_epoch: {best_epoch}')
    results = build_and_train_model(
        source_train_x, source_train_y, target_train_x, source_test_x, source_test_y, target_test_x, target_test_y,
        arch=arch, test_model=True, num_epochs=best_epoch, **params)
    src_stu_scores_dict = results[0]
    src_tea_scores_dict = results[1]
    tgt_stu_scores_dict = results[2]
    tgt_tea_scores_dict = results[3]
    training_time = results[4]
    inference_time = results[5]
    log(f'src_stu_scores_dict: {src_stu_scores_dict}')
    log(f'src_tea_scores_dict: {src_tea_scores_dict}')
    log(f'tgt_stu_scores_dict: {tgt_stu_scores_dict}')
    log(f'tgt_tea_scores_dict: {tgt_tea_scores_dict}')
    log(f'training_time: {training_time}')
    log(f'inference_time for 1000 instences: {inference_time}')
    tgt_tea_scores_dict.update({'training_time': training_time, 'inference_time': inference_time, 'params': params})
    tgt_scores = pd.Series(tgt_tea_scores_dict)
    return tgt_scores


if __name__ == '__main__':
    # The hyper-parameters that got in the paper
    # confidence_thresh = 0.96837722, teacher_alpha = 0.99, unsup_weight = 3.0, cls_balance = 0.005, learning_rate = 0.001
    global source_train_x, source_train_y, target_train_x, target_train_y

    source_x, source_y, target_x, target_y, n_classes = load_data(exp=exp)

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None

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

            domain_adapt_BO = BayesianOptimization(evaluate_exp, {'confidence_thresh': (0.5, 1.0),
                                                                'teacher_alpha': (0.5, 1.0),
                                                                'unsup_weight': (1.0, 3.0),
                                                                'cls_balance': (0.005, 0.01),
                                                                'learning_rate': (0.0001, 0.01)
                                                                })
            domain_adapt_BO.maximize(init_points=init_points, n_iter=bo_num_iter)
            params_domain_adapt = domain_adapt_BO.max['params']
            log(f'outer CV {cv_idx} - Opt hyper params: {params_domain_adapt} \n with target of: {domain_adapt_BO.max["target"]}')

            tgt_scores = rebuild_and_test_model(params_domain_adapt, source_train_x, source_train_y, target_train_x, source_test_x,
                               source_test_y, target_test_x, target_test_y)
            tgt_scores_list.append(tgt_scores)
            log('*********************************************************************** \n')
    except Exception:
        log(f"stopped at cv: {cv_idx}")
        raise
    finally:
        if len(tgt_scores_list) > 0:
            tgt_scores_df = pd.DataFrame(tgt_scores_list)
            date = datetime.datetime.now().strftime('%d-%m_%H-%M')
            tgt_scores_df.to_csv(f"./mean_teacher/tgt_scores_df_{exp}_until_cv_{cv_idx}_{date}.csv")
