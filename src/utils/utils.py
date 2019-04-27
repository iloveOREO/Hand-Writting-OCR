import torch 

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
    """ Get accuracy given output and targets
    """
    pred, _ = get_prediction(output, prob)
    cnt = 0
    for batch_ind, target in enumerate(targets):
        target = [v for v in target]
        if target == pred[batch_ind]:     
            cnt += 1
    return float(cnt) / len(targets)

class AverageMeter(object):
    """ Average meter.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset items.
        """
        self.n = 0
        self.val = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, n=1):
        """ Update
        """
        self.n += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.n
