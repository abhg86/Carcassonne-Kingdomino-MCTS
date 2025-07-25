import os
import sys
import time

import numpy as np
from tqdm import tqdm

from MCTS_util import dotdict

# sys.path.append('../../')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOARD_SIZE = 35


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



args_net = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

BOARD_SIZE = 35

class NNetWrapper():
    def __init__(self, args=args_net, board_size=BOARD_SIZE, player_nb=2):
        self.nnet = Carcassonne_NNet(args,player_nb, board_size)
        self.action_size = board_size*board_size*9 + 1
        self.board_size = board_size*3
        self.args = args                    # for copying in coach

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples,nb_train=144, numEps=64):
        """
        examples: list of examples, each example is of form (state, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            # batch_count = int(nb_train * numEps / self.args.batch_size)
            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                states, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    states, target_pis, target_vs = states.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(states)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), states.size(0))
                v_losses.update(l_v.item(), states.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, numpy_state):
        """
        numpy_state: np array with board, meeples, next tile, current player and phase
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(numpy_state.astype(np.float64))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1,-1, self.board_size, self.board_size)         # board_size =board_size*3 here
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])



class Carcassonne_NNet(nn.Module):
    def __init__(self, args, player_nb, board_size=BOARD_SIZE):
        # game params
        self.action_size = board_size*board_size*9 +1
        self.args = args
        self.in_channels = 5 + 5 + player_nb*4 + player_nb + 2
        self.board_size = board_size*3

        super(Carcassonne_NNet, self).__init__()
        self.conv1 = nn.Conv2d(self.in_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_size-4)*(self.board_size-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x in_channels x board_size*3 x board_size*3
        # s = s.view(-1, self.in_channels, self.board_size, self.board_size)                # batch_size x in_channels  x board_size*3 x board_size*3
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_size*3 x board_size*3
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_size*3 x board_size*3
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_size*3-2) x (board_size*3-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_size*3-4) x (board_size*3-4)
        s = s.view(-1, self.args.num_channels*(self.board_size-4)*(self.board_size-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)