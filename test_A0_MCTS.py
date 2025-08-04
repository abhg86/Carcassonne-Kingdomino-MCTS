import logging
from multiprocessing import set_start_method

import coloredlogs

from Coach import Coach
from A0_NNet import NNetWrapper as nn
from MCTS_util import *

import torch

import gc
import tracemalloc

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 40,
    'numEps': 64,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 1,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 2,

})

args_net = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 16,
    'cuda': torch.cuda.is_available(),
    'num_channels': 1,
})

BOARD_SIZE = 35

def main():
    log.info('Loading %s... Carcassonne')

    log.info('Loading %s...', nn.__name__)
    nnet = nn(args_net, BOARD_SIZE, player_nb=2)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(nnet, args, BOARD_SIZE)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Required for multiprocessing on some platforms
    main()