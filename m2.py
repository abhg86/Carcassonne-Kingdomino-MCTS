from multiprocessing import set_start_method
from m1 import f3

if __name__ == '__main__':
    # set_start_method('spawn', force=True)  # Required for multiprocessing on some platforms
    print(f3()) # this would run multiprocessing code
