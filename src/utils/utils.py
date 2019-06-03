import torch
from torch.autograd import Variable
import torch.nn.functional as F


def _is_tuple(tuple_like):
    return isinstance(tuple_like, tuple)


def preprocess_target(targets):
    """ 
    Preprocess targets.
        :param target: list of `torch.IntTensor`
    """
    lengths = [len(t) for t in targets]
    lengths = torch.IntTensor(lengths)

    flatten_target = torch.cat([torch.Tensor(t) for t in targets])
    return flatten_target, lengths


def get_seq_length(x):
    """ 
    Get sequence lengths of batch of data
        :param x: batch data
    """
    bsz, length = x.size(0), x.size(3)
    lengths = torch.IntTensor(bsz).fill_(length)
    return lengths


def tensor_to_variable(tensor, volatile):
    if _is_tuple(tensor):
        return tuple((tensor_to_variable(x, volatile) for x in tensor))
    else:
        return Variable(tensor, volatile)


def get_accuracy(output, targets, prob=True):
    """ 
    Get accuracy given output and targets
    """
    pred, _ = get_prediction(output, prob)
    cnt = 0
    for batch_ind, target in enumerate(targets):
        target = [v for v in target]
        if target == pred[batch_ind]:
            cnt += 1
    return float(cnt) / len(targets)


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
