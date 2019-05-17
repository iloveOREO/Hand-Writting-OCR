import torch
import warpctc_pytorch
from warpctc_pytorch import CTCLoss
ctc_loss = CTCLoss()
# ctc_loss = warpctc_pytorch.gpu_ctc
print(ctc_loss)
# expected shape of seqLength x batchSize x alphabet_size
probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()

labels = torch.IntTensor([1, 2])
# labels = labels.to("cuda:0")
label_sizes = torch.IntTensor([2])
probs_sizes = torch.IntTensor([2])
probs.requires_grad_(True)  # tells autograd to compute gradients for probs
cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
print(cost)
probs = probs.to("cuda:0")
cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
print(cost)
# cost.backward()