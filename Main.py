from opts import parser
from Dataset import VideoDataset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from Models import Net
import torch.backends.cudnn as cudnn
import time
import os
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

def main():
    global args, best_prec1
    cudnn.benchmark = True
    args = parser.parse_args()

    if not os.path.exists (args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    strat_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = open(os.path.join(args.log_dir, strat_time + '.txt'), 'w')
    print (args.description)
    log.write(args.description + '\n')
    log.flush()
    print ('=======================Experimental Settings=======================\n')
    log.write('=======================Experimental Settings=======================\n')
    log.flush()
    print ('Using_Dataset:{0}  Batch_Size:{1}  Epochs:{2} '.format(args.dataset, args.batch_size, args.epoch))
    log.write('Using_Dataset:{0}  Batch_Size:{1}  Epochs:{2}'.format(args.dataset, args.batch_size, args.epoch) + '\n')
    log.flush()
    print ('Num_segments:{0}  Num_frames:{1}  Base_model:{2}\n'.format(args.segments, args.frames, args.base_model))
    log.write('Num_segments:{0}  Num_frames:{1}  Base_model:{2}\n'.format(args.segments, args.frames, args.base_model) + '\n')
    log.flush()
    print ('===================================================================\n')
    log.write('===================================================================\n')
    log.flush()


    test_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.test_video_list, num_segments=args.segments,
                     num_frames=args.frames, test_mode=True,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers*2, pin_memory=True, drop_last=True)

    net = Net(basemodel=args.base_model, num_segments=args.segments, num_frames=args.frames, dataset=args.dataset, d_model=args.d_model, start=args.start)
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load('./model/2018-08-18 08_16_57.pkl'))

    prec1 = test(test_loader, net)
    print('The testing accuracy is: {0}'.format(prec1))



def test(test_loader, net):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for input, target in tqdm(test_loader):
        input = input.squeeze(0).transpose(0, 1).contiguous()
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda()

        output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prediction = torch.max(output)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1)
        top5.update(prec5)

    return top1.avg


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:,:k].float().sum()
        store.append(corrrect_k * 100.0 / batch_size)
    return store


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr, lr_step):
    lr = lr * 0.1 ** (epoch // lr_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
