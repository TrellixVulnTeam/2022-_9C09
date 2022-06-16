#coding:utf-8
import argparse
from utils.utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader.get_loader import get_loader
import numpy as np
import os
import shutil

max_total = [0, 0, 0]

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--gpu', type=str, default='0', metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lamda', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--c_lr_rate', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--net', type=str, default='resnet50_ori', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--opt', type=str, default='SGD', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--beta', type=float, default=0.75, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--task_name', type=str, default='A_W_0.6_OSBP', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model or not')
parser.add_argument('--class_num', type=int, default=9, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save_path', type=str, default='checkpoint/checkpoint', metavar='B',
                    help='checkpoint path')
parser.add_argument('--source_path', type=str, default='./office31_label/amazon_source0.6.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--target_path', type=str, default='./office31_label/dslr_target0.6.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--root_path', type=str, default='/home/chenziliang/GNCD/Office31/', metavar='B',
                    help='checkpoint path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--unit_size', type=int, default=256, metavar='N',
                    help='unit size of fully connected layer')
parser.add_argument('--update_lower', action='store_true', default=False,
                    help='update lower layer or not')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.task_name = args.task_name + str(args.lamda) + '_' + str(args.beta) + '_' + str(args.lr) + '_' + str(
    args.c_lr_rate)
source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path
batch_size = args.batch_size

data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# if not os.path.exists(args.task_name+'.txt'):
#    os.mkdir(args.task_name+'.txt')
record_file = open('./result/' + args.task_name + '.txt', 'w')
use_gpu = torch.cuda.is_available()
# torch.manual_seed(task_num)
# if args.cuda:
#     torch.cuda.manual_seed(task_num)
train_loader, test_loader = get_loader(source_data, target_data, args.root_path, evaluation_data,
                                       data_transforms, batch_size=args.batch_size)
dataset_train = train_loader.load_data()
dataset_test = test_loader


def loss_entropy_no_softmax(input):
    loss = 0
    '''for i in range(input.size()[0]):
        soft_max = F.softmax(input[i])
        loss += -1.0*torch.dot(soft_max,torch.log(soft_max))'''
    loss = -1.0 * torch.dot(input.view(-1), torch.log(input + 1e-20).view(-1))
    loss /= input.size()[0]
    return loss

def loss_entropy(input):
    loss = torch.zeros(input.size()[0])
    for i in range(input.size()[0]):
        soft_max = F.softmax(input[i])
        tmp_loss = (-1.0*torch.dot(soft_max, torch.log(soft_max))).data.cpu()
        loss[i] = tmp_loss[0]
    return loss

num_class = args.class_num
# class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
# ['airplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle',
# 'person', 'plant', 'skateboard', 'train', 'truck', 'unk']
class_list = ["back_pack", "bike", "calculator", "headphone", "keyboard", "laptop",
              "monitor", "mouse", "mug", "projector", "unk"]
G, C = get_model(args.net, num_class=num_class, unit_size=args.unit_size)


if args.cuda:
    G.cuda()
    C.cuda()
opt_c, opt_g = get_optimizer_visda_ori(args.lr, G, C, args.c_lr_rate, args.update_lower, args.opt)
gamma = 0.001
power = 0.75


def train(num_epoch, max_total):
    criterion = nn.CrossEntropyLoss().cuda()
    loss_domain = torch.nn.BCEWithLogitsLoss().cuda()
    i = 0
    print('train start!')
    for ep in range(num_epoch):
        G.train()
        C.train()
        if ep > 0 and ep % 5 == 0:
            max_total = test(max_total)
            G.train()
            C.train()
        for batch_idx, data in enumerate(dataset_train):
            i += 1

            if i % 1000 == 0:
                print('iteration %d', i)

            img_s = data['S']
            label_s = data['S_label']
            img_t = data['T']
            label_t = data['T_label']
            if args.cuda:
                img_s, label_s = Variable(img_s.cuda()), \
                                 Variable(label_s.cuda())
                img_t = Variable(img_t.cuda())
            if len(img_t) < batch_size:
                break
            if len(img_s) < batch_size:
                break

            p = min(1.0, float(ep * 1.0) / (1 * num_epoch))
            alpha = float(2. / (1. + np.exp(-10 * p)) - 1)

            opt_g.zero_grad()
            opt_c.zero_grad()
            feat = G(img_s)
            out_s = C(feat)
            loss_s = criterion(out_s, label_s)
            loss_s.backward()
            target_funk = Variable(torch.FloatTensor(img_t.size()[0], 2).fill_(0.5).cuda())
            p = 1.0
            C.set_lambda(p)
            feat_t = G(img_t)
            out_t = C(feat_t, reverse=True)
            out_t = F.softmax(out_t)
            prob1 = torch.sum(out_t[:, :num_class - 1], 1).view(-1, 1)
            prob2 = out_t[:, num_class - 1].contiguous().view(-1, 1)

            prob = torch.cat((prob1, prob2), 1)
            loss_t = bce_loss(prob, target_funk)
            loss_t.backward()
            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()

            if batch_idx % args.log_interval == 0:
                np.set_printoptions(precision=2)
                print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss Source: {:.6f}\t Loss Target: {:.6f}'.format(
                    ep, batch_idx * (img_s.size()[0]), 70000,
                        100. * batch_idx / 70000, loss_s.data[0], loss_t.data[0]))


        if args.save:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            save_model(G, C, args.save_path + '_' + str(ep))
    return max_total


def test(max_total):
    G.eval()
    C.eval()
    correct = 0
    size = 0
    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    for batch_idx, data in enumerate(dataset_test):
        if args.cuda:
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                             Variable(label_t.cuda(), volatile=True)
        feat = G(img_t)
        out_t = C(feat)
        pred = out_t.data.max(1)[1]
        k = label_t.data.size()[0]
        correct += pred.eq(label_t.data).cpu().sum()
        pred = pred.cpu().numpy()
        for t in range(num_class):
            t_ind = np.where(label_t.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
        size += k
    per_class_acc = per_class_correct / per_class_num

    os_acc = float(per_class_acc.mean())
    oss_acc = float(per_class_acc[:-1].mean())
    unk = float(per_class_acc[-1])
    if (unk + oss_acc) > max_total[2]+max_total[1]:#大于最高纪录
        max_total = [os_acc, oss_acc , unk]
    torch.save(G.state_dict(), os.path.join('./snapshot', args.task_name + "_max_G.pth"))
    torch.save(C.state_dict(), os.path.join('./snapshot', args.task_name + "_max_C.pth"))
    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} (os:{:.2f}%)  (oss:{:.2f}%)\n'.format(
            correct, size,
            100. * os_acc, 100. * oss_acc))
    record_file.write(
        '\nTest set including unknown classes:  Accuracy: {}/{} (os:{:.2f}%)  (oss:{:.2f}%)\n'.format(
            correct, size,
            100. * os_acc, 100. * oss_acc))
    print('UNK:%s' % ( per_class_acc[-1]))
    record_file.write('UNK:%s' % (per_class_acc[-1]))
    return max_total


import time
start_time = time.time()
max_total = train(args.epochs + 1, max_total)
print('max acc :[ {:.4f}%,{:.4f}%]'.format(max_total[0], max_total[1]))
record_file.write('max acc :[ {:.4f}%,{:.4f}%]'.format(max_total[0], max_total[1]))
record_file.close()

file = open('final_result.txt', 'a')
file.write(args.task_name + '\n')
file.write('os oss unk : {:.4f}\t{:.4f}\t{:.4f}]'.format(max_total[0]*100.0, max_total[1]*100.0, max_total[2]*100.0)+ '\n')
file.close()

shutil.copy(os.path.join('./snapshot', args.task_name + "_max_G.pth"),
            os.path.join('./snapshot', args.task_name + "_G_" + str(max_total[0] * 100)[:5] + ".pth"))
shutil.copy(os.path.join('./snapshot', args.task_name + "_max_C.pth"),
            os.path.join('./snapshot', args.task_name + "_C_" + str(max_total[0] * 100)[:5] + ".pth"))
end_time = time.time()

print('total time:')
print(end_time-start_time)
