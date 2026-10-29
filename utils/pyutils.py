import logging
import datetime
import numpy as np

from texttable import Texttable
import torch
import random


def setup_logger(filename='test.log'):

    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

    def results(self):
        results = {}
        for k in self.__data.keys():
            v = self.get(k)
            results[k] = v
        return results

def fix_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def crf_inference_label(image, mask, n_labels, t=10, gt_prob=0.7):
    """
    `n_labels`: DO include `0` in counting labels.
    `t`: run `t` inference steps.

    return `result`: (H,W), processed by argmax.
    """
    from pydensecrf.utils import unary_from_labels
    import pydensecrf.densecrf as dcrf

    h, w = image.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(mask, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)

    ### default sxy: 80, default srgb 13
    ### The names ('sxy' and 'srgb') are shorthand for "x/y standard-deviation" and "rgb standard-deviation"
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=image, compat=10)

    q = d.inference(t)

    return np.argmax(q, axis=0).reshape((h, w))


class ListMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = []

    def add(self, dict_):
        for k, v in dict_.items():
            if k not in self.__data:
                self.__data[k] = []
            self.__data[k].append(v)

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]]
        else:
            v_list = [self.__data[k] for k in keys]
            return tuple(v_list)

    def get_mean(self, *keys):
        if len(keys) == 1:
            return np.mean(self.__data[keys[0]])
        else:
            v_list = [np.mean(self.__data[k]) for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = []
        else:
            v = self.get(key)
            self.__data[key] = []
            return v


def multi_label_accuracy(outputs, targets):
    """
    outputs: B, C
    targets: B, C
    """
    if len(outputs.shape) == 1:
        outputs = torch.unsqueeze(outputs, dim=0)
    outputs = outputs.sigmoid() > 0.5
    correct = ((outputs == targets) & (targets == 1)).sum().item()
    total = outputs.sum().item()
    true_total = (targets == 1).sum().item()
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0
    if true_total > 0:
        recall = correct / true_total
    else:
        recall = 0
    return accuracy, recall
