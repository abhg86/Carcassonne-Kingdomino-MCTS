from multiprocessing import Pool 
import multiprocessing as mp

import torch

from A0_NNet import NNetWrapper as A0_NNet
from Coach import Coach
from MCTS_util import dotdict   

args_net = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 2,
    'cuda': torch.cuda.is_available(),
    'num_channels': 1,
})

args = dotdict({
    'numIters': 1,
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 1,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def execc(coach:Coach,q):
    q.put(coach.executeEpisode())

def f(x,q):
    q.put(x**2)

def f2():
    # p = Pool(5)
    # with Pool(processes=5) as p:
    #     # Use the pool to map the function f to a list of numbers
    #     p = [p.apply_async(f, (i,)) for i in range(10)]
    #     p = [res.get() for res in p]
    q = mp.Queue()
    processes = []
    for i in range(2):
        print(f"Process {i} is running")
        p = mp.Process(target=execc, args=(Coach(A0_NNet(args_net), args),q,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    res = [q.get() for _ in range(2)]
    return res

def f3():
    q = mp.Queue()
    for i in range(2):
        execc(Coach(A0_NNet(args_net), args),q)
    return [q.get() for _ in range(2)]