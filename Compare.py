import logging 
from random import choice
import numpy as np
from A0_MCTS import A0_MCTS
from heuristic_score_MCTS import Sc_MCTS
import torch

from A0_NNet import NNetWrapper
from Arena import Arena
from MCTS_util import dotdict, number_to_action
from wingedsheep.carcassonne.utils.action_util import ActionUtil

# log = logging.getLogger(__name__)

BOARD_SIZE = 35

args = dotdict({
    'numIters': 100,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 72,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

args_net = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 1,
})

# nnet = NNetWrapper(args_net, BOARD_SIZE, player_nb=2)
# nnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
# mcts = A0_MCTS(nnet, args.numMCTSSims, args.cpuct,board_size=BOARD_SIZE)
mcts = Sc_MCTS(args.numMCTSSims, args.cpuct,board_size=BOARD_SIZE)


# log.info('PITTING AGAINST PREVIOUS VERSION')
arena = Arena(lambda x: number_to_action(np.argmax(mcts.getActionProb(x, temp=0, num_plays=3)), x.phase, x.next_tile, board_size=BOARD_SIZE),
                lambda x: choice(ActionUtil.get_possible_actions(x)))
wins1, wins2, draws = arena.playGames(args.arenaCompare)

print(f"wins1: {wins1}, wins2: {wins2}, draws: {draws}")

