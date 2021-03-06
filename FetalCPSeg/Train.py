import os
import time

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable

from DataOp import TrainGenerator, get_list
from Network import MixAttNet
from Utils import check_dir, AvgMeter, dice_score


def plot_metric_per_epoch(num_iterations, metric_epoch, tag):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # zip(*dice_epoch)
    # plt.scatter(*zip(*avg_dice_total), color='red')
    plt.scatter(num_iterations, metric_epoch, color='red')
    ax.set_facecolor('gray')
    if tag == 'dice_coeff':
        plt.title('Dice coefficient per epoch')
        plt.ylabel('Dice coefficient')
    elif tag == 'loss':
        plt.title('Loss per epoch')
        plt.ylabel('Loss')
    elif tag == 'accuracy':
        plt.title('Accuracy per epoch')
        plt.ylabel('Accuracy')
    elif tag == 'error_rate':
        plt.title('Error per epoch')
        plt.ylabel('Error')

    plt.grid(b=True, color='white')
    plt.xlabel('Iterations')
    # for i,j in *zip(*avg_dice_total):
    #    ax.annotate(str(j),xy=(i,j))
    plt.show()


def adjust_lr(optimizer, iteration, num_iteration):
    """
    we decay the learning rate by a factor of 0.1 in 1/2 and 3/4 of whole training process
    """
    if iteration == num_iteration // 2:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif iteration == num_iteration // 4 * 3:
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


def loss_func(predict, label, pos_weight):
    """
    here we define the loss function, which you can upload additional loss in here
    """
    bce_loss = F.binary_cross_entropy_with_logits(predict, label, pos_weight=pos_weight)
    return bce_loss


def train_batch(net, optimizer, loader, patch_size, batch_size):
    net.train()

    image, label = loader.get_item()
    # here we calculate the positive ratio in the input batch data
    if np.where(label == 1)[0].shape[0] == 0:
        weight = 1
    else:
        weight = batch_size*patch_size*patch_size*patch_size/np.where(label == 1)[0].shape[0]

    image = Variable(torch.from_numpy(image).cuda())
    label = Variable(torch.from_numpy(label).cuda())
    
    predict = net(image)

    optimizer.zero_grad()

    weight = torch.FloatTensor([weight]).cuda()
    loss1 = loss_func(predict[0], label, pos_weight=weight)
    loss2 = loss_func(predict[1], label, pos_weight=weight)
    loss3 = loss_func(predict[2], label, pos_weight=weight)
    loss4 = loss_func(predict[3], label, pos_weight=weight)
    loss5 = loss_func(predict[4], label, pos_weight=weight)
    loss6 = loss_func(predict[5], label, pos_weight=weight)
    loss7 = loss_func(predict[6], label, pos_weight=weight)
    loss8 = loss_func(predict[7], label, pos_weight=weight)
    loss9 = loss_func(predict[8], label, pos_weight=weight)
    loss = loss1 + \
           0.8*loss2 + 0.7*loss3 + 0.6*loss4 + 0.5*loss5 + \
           0.8*loss6 + 0.7*loss7 + 0.6*loss8 + 0.5*loss9
    #loss = loss_func(predict, label, pos_weight=weight)

    loss.backward()
    optimizer.step()
    return loss.item()


def val(net, val_list, patch_size):
    net.eval()
    metric_meter = AvgMeter()

    for data_dict in val_list:

        image_path = data_dict['image_path']
        label_path = data_dict['label_path']
        image_ = nib.load(image_path).get_fdata()
        label_ = nib.load(label_path).get_fdata()
        image = TrainGenerator.preprocess(image_, np.nanmin(image_)).squeeze()
        label = TrainGenerator.preprocess(label_, 0).squeeze()
        pre_count = np.zeros_like(image, dtype=np.float32)
        predict = np.zeros_like(image, dtype=np.float32)

        w, h = image.shape
        x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size // 4)[:, np.newaxis],
                                            np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
        y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size // 4)[:, np.newaxis],
                                            np.array([h - patch_size])[:, np.newaxis])).astype(np.int))

        for x in x_list:
            for y in y_list:
                image_patch = image[x:x+patch_size, y:y+patch_size].astype(np.float32)
                image_patch_tensor = torch.from_numpy(image_patch[np.newaxis, np.newaxis]).cuda()
                pre_patch = net(image_patch_tensor).squeeze()
                predict[x:x+patch_size, y:y+patch_size] += pre_patch.cpu().data.numpy()
                pre_count[x:x+patch_size, y:y+patch_size] += 1
        predict /= pre_count
        metric_meter.update(dice_score(predict, label))

    return metric_meter.avg


def main(args):
    #torch.cuda.set_device(args.gpu_id)

    check_dir(args.output_path)
    ckpt_path = os.path.join(args.output_path, "ckpt")
    check_dir(ckpt_path)

    train_list, val_list, test_list = get_list(args.data_path, args.mask_path)

    train_generator = TrainGenerator(train_list,
                                     batch_size=args.batch_size,
                                     patch_size=args.patch_size)
    net = MixAttNet().cuda()

    # Check number of parameters of the network
    params = 0
    for parameter in net.parameters():
        params += 1

    print('PARAMETERS: '+str(params))
    
    # Use a dictionary already optimised 
    #net.load_state_dict(torch.load('pretrained_val_nothing.pth.gz'))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    open(os.path.join(args.output_path, "train_record.txt"), 'w+')

    loss_meter = AvgMeter()
    start_time = time.time()
    best_metric = 0.

    for iteration in range(1, args.num_iteration+1):
        adjust_lr(optimizer, iteration, args.num_iteration)
        train_loss = train_batch(net=net, optimizer=optimizer, loader=train_generator, patch_size=args.patch_size, batch_size=args.batch_size)
        loss_meter.update(train_loss)

        if iteration % args.pre_fre == 0:
            print('Iteration1: ' + str(iteration))
            iteration_time = time.time() - start_time
            info = [iteration, loss_meter.avg, iteration_time]
            print("Iter[{}] | Loss: {:.3f} | Time: {:.2f}".format(*info))
            start_time = time.time()
            loss_meter.reset()

        if iteration % args.val_fre == 0:
            print('Iteration2: ' + str(iteration))
            val_dice = val(net, test_list, args.patch_size)
            if val_dice > best_metric:
                print('Best Metric: ' + str(val_dice))
                torch.save(net.state_dict(), os.path.join(ckpt_path, "best_val.pth.gz"))
                best_metric = val_dice
            open(os.path.join(args.output_path, "train_record.txt"), 'a+').write("{:.3f} | {:.3f}\n".format(train_loss, val_dice))
            print("Val in Iter[{}] Dice: {:.3f}".format(iteration, val_dice))
        if iteration % 100 == 0:
            print('Iteration3: ' + str(iteration))
            torch.save(net.state_dict(), os.path.join(ckpt_path, "train_{}.pth.gz".format(iteration)))


if __name__ == '__main__':

    class Parser(object):
        def __init__(self):
            self.gpu_id = 0

            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.batch_size = 32

            self.num_iteration = 60000
            self.val_fre = 200
            self.pre_fre = 20

            self.patch_size = 64

            self.data_path = '../Data/imgs_2D/'
            self.mask_path = '../Data/mask_2D/'
            self.output_path = 'output_b32_nothing_test/'

    parser = Parser()
    main(parser)
