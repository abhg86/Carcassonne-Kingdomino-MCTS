from multiprocessing import set_start_method
from m1 import f2, f3

if __name__ == '__main__':
    set_start_method('spawn', force=True)  # Required for multiprocessing on some platforms
    print(f2()[-1][-1][-1]) # this would run multiprocessing code
