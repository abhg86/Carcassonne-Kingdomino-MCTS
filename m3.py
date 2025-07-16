import torch.multiprocessing as mp
from A0_NNet import NNetWrapper as MyModel
from MCTS_util import dotdict

def train(dic, q):
    # # Construct data_loader, optimizer, etc.
    # for data, labels in data_loader:
    #     optimizer.zero_grad()
    #     loss_fn(model(data), labels).backward()
    #     optimizer.step()  # This will update the shared parameters
    q.put(dic['lr'])  # Put the result in the queue

dic = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 2,
    'num_channels': 1,
})

if __name__ == '__main__':
    num_processes = 4
    # NOTE: this is required for the ``fork`` method to work
    # model.share_memory()
    processes = []

    q = mp.Queue()
    for rank in range(num_processes):

        p = mp.Process(target=train, args=(dic,q,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    print([q.get() for _ in range(num_processes)])  # Get results from the queue
