import os
import time
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss, Loss_CE, Loss_triplet, Reweighted_Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from utils.per_sample_loss import plot_loss_distribution

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.tensorboard import SummaryWriter 

class Main():
    def __init__(self, model, loss, data, writer):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.real_loader = data.real_train_loader
        self.fake_loader = data.fake_train_loader

        self.model = model.to('cuda')
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        self.writer = writer

    def train(self, epo_idx):

        # self.scheduler.step()

        self.model.train()

        num_samples = len(self.train_loader)

        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            self.writer.add_scalar('loss', loss.item(), batch + epo_idx * num_samples)
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        
    def persample_loss(self, num_batch=-1, save_path=''):
        self.model.eval()

        loss_list = []

        if num_batch == -1:
            num_batch = len(self.real_loader)

        for batch, (inputs, labels) in enumerate(self.real_loader):
            if batch > num_batch:
                break
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss_list.append(loss.item())

        for batch, (inputs, labels) in enumerate(self.fake_loader):
            if batch > num_batch:
                break
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss_list.append(loss.item())

        lower_bound = min(loss_list)
        loss_range = max(loss_list) - lower_bound
        
        loss_list = [(x - lower_bound) / loss_range for x in loss_list]

        acc = plot_loss_distribution(real_loss=loss_list[:num_batch], fake_loss=loss_list[num_batch:], save_path=save_path)

        print(acc)


    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data()
    model = MGN()
    # loss = Loss()
    loss = Reweighted_Loss()
    writer = SummaryWriter('./logs/' + str(opt.fake_ratio) + time.asctime(time.localtime(time.time())))
    main = Main(model, loss, data, writer)

    if opt.mode == 'train':
        os.makedirs('weights', exist_ok=True)
        os.makedirs('train_loss/{}fake'.format(opt.fake_ratio), exist_ok=True)

        for epoch in range(1, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train(epoch - 1)
            for name, param in model.named_parameters():
                writer.add_histogram(name + '_grad', param.grad, epoch)
                writer.add_histogram(name + '_data', param, epoch)

            if epoch % 5 == 0:
                print('plot per sample loss')
                main.persample_loss(num_batch=-1, save_path='train_loss/rew_{}fake/loss{}.png'.format(opt.fake_ratio, epoch))
            
            if epoch % 20 == 0:
                print('\nstart evaluate')
                main.evaluate()
                torch.save(model.state_dict(), ('weights/rew_{}fake/model_{}{}.pt'.format(opt.fake_ratio, epoch, opt.fake_ratio)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'plot':
        print('plot per sample loss')
        os.makedirs('plot_loss/0.0fake', exist_ok=True)
        model.load_state_dict(torch.load(opt.weight))
        main.persample_loss(num_batch=-1, save_path='plot_loss/0.0fake/loss.png')

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
