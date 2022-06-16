# -*- coding: utf-8 -*-
import torch.optim as opt
from models.basenet import *
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from modules.module import feat2prob, target_distribution
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def init_prob_kmeans(model, eval_loader, args):
    torch.manual_seed(1)
    model = model.cuda()
    # cluster parameter initiate
    model.eval()
    feats = np.zeros((len(eval_loader.dataset), 2048))
    targets = np.zeros(len(eval_loader.dataset))
    for _, data in enumerate(eval_loader):
        img_t, label_t, idx = data[0], data[1], data[2]
        img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                         Variable(label_t.cuda(), volatile=True)
        feat = model(img_t)
        feat = feat.view(img_t.size(0), -1)
        idx = idx.numpy()
        feats[idx, :] = feat.data.cpu().numpy()
        targets[idx] = label_t.data.cpu().numpy()

    # evaluate clustering performance
    # 先PCA降维
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)

    # 初始化聚类中心
    init_center = np.empty((args.n_clusters, feats.shape[1]))
    for c in range(args.n_clusters):
        init_center[c] = feats[targets == 0].mean(0)

    # 使用kmeans聚类初始化中心点
    kmeans = KMeans(n_clusters=args.n_clusters, init=init_center, n_init=20)
    y_pred = kmeans.fit_predict(feats)
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs


def kmeans_cluster(model, C, eval_loader, args, out_label_path):
    torch.manual_seed(1)
    model = model.cuda()
    # cluster parameter initiate
    model.eval()
    C.eval()
    feats = np.zeros((len(eval_loader.dataset), 2048))
    targets = np.zeros(len(eval_loader.dataset))
    cls_score = np.zeros(len(eval_loader.dataset))
    for _, data in enumerate(eval_loader):
        img_t, label_t, idx = data[0], data[1], data[2]
        img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                         Variable(label_t.cuda(), volatile=True)
        feat = model(img_t)
        out = C(feat)
        cls_score[idx] = F.softmax(out).data.max(1)[0].cpu().numpy()
        feat = feat.view(img_t.size(0), -1)
        idx = idx.numpy()
        feats[idx, :] = feat.data.cpu().numpy()
        targets[idx] = label_t.data.cpu().numpy()

    # evaluate clustering performance
    # 先PCA降维
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)

    # 初始化聚类中心
    init_center = np.empty((args.n_clusters, feats.shape[1]))
    for c in range(args.n_clusters):
        if (targets == c).max() == True:
            init_center[c] = feats[targets == c].mean(0)
        else:
            init_center[c] = np.empty((args.n_clusters, feats.shape[1]))[0]
            # 使用kmeans聚类初始化中心点
    kmeans = KMeans(n_clusters=args.n_clusters, init=init_center, n_init=20)
    y_pred = kmeans.fit_predict(feats)

    u_idx = y_pred > 9
    u_pred = y_pred[u_idx]
    u_target = targets[u_idx]
    u_target[u_target < 10] = -1

    acc, nmi, ari = cluster_acc_with_noise(u_target, u_pred), nmi_score(u_target, u_pred), ari_score(u_target, u_pred)
    print('unkonwn cluster acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    # 筛选高置信度标签
    k_high_confi_idx = cls_score > 0.9
    uk_high_confi_idx = cls_score > 0.80
    k_idx = targets < 10
    uk_idx = targets > 9
    high_confi_idx = k_high_confi_idx*k_idx + uk_high_confi_idx*uk_idx

    # 输出高置信度标签可靠性
    k_all = 0
    k_acc = 0
    uk_all = 0
    uk_acc = 0
    for i in range(high_confi_idx.shape[0]):
        if high_confi_idx[i] == True:
            if y_pred[i] < 10:
                k_all += 1
                if targets[i] == y_pred[i]:
                    k_acc += 1
            else:
                uk_all += 1
                if targets[i] > 9:
                    uk_acc += 1
    print('knonws', k_acc, k_all, k_acc * 1.0 / k_all)
    print('unknonws', uk_acc, uk_all, uk_acc * 1.0 / uk_all)

    lines = open(args.target_path, 'r').readlines()
    f_out = open(out_label_path, 'w')
    for i in range(len(lines)):
        path, _ = lines[i].strip().split(' ')
        if high_confi_idx[i] == True:
            f_out.write(path + ' ' + str(y_pred[i]) + '\n')

    return acc, nmi, ari, kmeans.cluster_centers_


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def cluster_acc_with_noise(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`,-1 indicate noisy label
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    all_count = y_pred.size
    for i in range(y_true.shape[0] - 1, -1, -1):
        if y_true[i] == -1:
            y_true = np.delete(y_true, i)
            y_pred = np.delete(y_pred, i)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / all_count


def get_model(net, num_class=13, unit_size=100, domain_classifier=False):
    if domain_classifier:
        if net == 'alex':
            model_g = AlexBase()
            model_c = Classifier(num_classes=num_class)
        elif net == 'vgg':
            model_g = VGGBase()
            model_c = Classifier(num_classes=num_class)
        elif 'ori' in net:
            model_g = ResBase_ori(net, unit_size=unit_size)
            model_c = ResClassifier_ori(num_classes=num_class, unit_size=unit_size)
            model_d = ResDiscriminator(unit_size=1024)
        else:
            model_g = ResBase(net, unit_size=unit_size)
            model_c = ResClassifier(num_classes=num_class, unit_size=unit_size)
            model_d = ResDiscriminator(unit_size=1024)
        return model_g, model_c, model_d
    else:
        if net == 'alex':
            model_g = AlexBase()
            model_c = Classifier(num_classes=num_class)
        elif net == 'vgg_fea':
            model_g = VGGBase_fea()
            model_c = Classifier(num_classes=num_class)
        elif net == 'vgg':
            model_g = VGGBase()
            model_c = Classifier(num_classes=num_class)
        elif 'ori' in net:
            model_g = ResBase_ori(net, unit_size=unit_size)
            model_c = ResClassifier_ori(num_classes=num_class, unit_size=unit_size)
        elif 'fea' in net:
            model_g = ResBase_fea(net, unit_size=unit_size)
            model_c = ResClassifier(num_classes=num_class, unit_size=unit_size)
        else:
            model_g = ResBase(net, unit_size=unit_size)
            model_c = ResClassifier(num_classes=num_class, unit_size=unit_size)
        return model_g, model_c


def get_optimizer_visda(lr, G, C, c_lr_rate=1.0, update_lower=False, opt_method='SGD'):
    if not update_lower:
        params = G.parameters()
        # params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
        #     G.bn1.parameters()) + list(G.bn2.parameters())) #+ list(G.bn4.parameters()) + list(
        # G.bn3.parameters()) + list(G.linear3.parameters()) + list(G.linear4.parameters()))
    else:
        params = G.parameters()
    if opt_method == 'SGD':
        optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr * c_lr_rate,
                              weight_decay=0.0005, nesterov=True)
    else:
        optimizer_g = opt.Adam(params, lr=lr, betas=(0.9, 0.999))
        optimizer_c = opt.Adam(list(C.parameters()), lr=lr * c_lr_rate, betas=(0.9, 0.999))
    return optimizer_g, optimizer_c


def get_optimizer_visda_ori(lr, G, C, c_lr_rate=1.0, update_lower=False, opt_method='SGD'):
    if not update_lower:
        params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
            G.bn1.parameters()) + list(G.bn2.parameters()) )# + list(G.bn4.parameters()) + list(
            # G.bn3.parameters()) + list(G.linear3.parameters()) + list(G.linear4.parameters()))
    else:
        params = G.parameters()
    if opt_method == 'SGD':
        optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr * c_lr_rate,
                              weight_decay=0.0005, nesterov=True)
    else:
        optimizer_g = opt.Adam(params, lr=lr, betas=(0.9, 0.999))
        optimizer_c = opt.Adam(list(C.parameters()), lr=lr * c_lr_rate, betas=(0.9, 0.999))
    return optimizer_g, optimizer_c


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c


def adjust_learning_rate(optimizer, lr, batch_id, max_id, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    beta = 0.75
    alpha = 10
    p = min(1, (batch_id + max_id * epoch) / float(max_id * max_epoch))
    lr = lr / (1 + alpha * p) ** (beta)  # min(1, 2 - epoch/float(20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
