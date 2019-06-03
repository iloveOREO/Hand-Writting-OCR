import os
import torch
import torch.optim as optim

from warpctc_pytorch import CTCLoss
from utils.utils import *
from model.DenseNet import *
from model.read_data import *
from model.Net import CRNN
from TextDataset import *
import torch.backends.cudnn as cudnn

if __name__ == '__main__':

    model = CRNN(81)
    # criterion = CTCLoss()
    criterion = CTCLoss()
    solver = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    ocr_dataset = TextDataset("./data", transform)
    ocr_dataset_loader = torch.utils.data.DataLoader(dataset=ocr_dataset,
                                                     batch_size=5,
                                                     shuffle=False,
                                                     collate_fn=Collate())

    use_cuda = torch.cuda.is_available()
    # loss_meter = AverageMeter()
    # acc_meter = AverageMeter()
    if use_cuda:
        # cudnn.benchmark = True
        cudnn.benchmark = True
        device = torch.device('cuda:0')
        model.cuda()
        criterion = criterion.cuda()
    for ind, (x, target, target_lengths) in enumerate(ocr_dataset_loader):
        # x = [batch_size, channels, height, weight]
        # act_lengths = get_seq_length(x)
        # target is a list of `torch.InTensor` with `bsz` size.
        '''
        flatten_target, target_lengths = preprocess_target(target)
        x, flatten_target, target_lengths = tensor_to_variable(
            (x, flatten_target, target_lengths), volatile=False)
        '''
        if use_cuda:
            x = x.cuda()
            # act_lengths = act_lengths.to(device)
            # flatten_target = flatten_target.to(device)
            # target_lengths = target_lengths.to(device)

        bsz = x.size(0)

        output = model(x)
        act_lengths = torch.IntTensor([output.size(0)] * 5)
        #  print(output.shape)
        loss = criterion(output, target, act_lengths, target_lengths)
        print(loss)
        solver.zero_grad()
        loss.backward()
        #         nn.utils.clip_grad_norm(model.parameters(), 10)
        solver.step()

        # loss_meter.update(loss.item())

        # acc = get_accuracy(output, target)
        # acc_meter.update(acc)
        # print(loss_meter)
        # print(acc_meter)
        # if use_tensorboard:
        #     logger.add_scalar('train_acc', acc)
        #
        # if (ind+1) % 100 == 0 or (ind+1) == len(training_loader):
        #     print('train:\t[{:03d}/{:03d}],\t'
        #           '[{:02d}/{:02d}]\t'
        #           'loss: {loss.avg:.4f}({loss.val:.4f})\t'
        #           'accuracy: {acc.avg:.4f}({acc.val:.4f})'.format(epoch, max_epoch,
        #           ind+1, len(training_loader), loss=loss_meter, acc=acc_meter))
        # break