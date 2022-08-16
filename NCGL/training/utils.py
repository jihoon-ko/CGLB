import random
import numpy as np
import torch
import dgl
from random import sample
import os
import errno

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)

def remove_illegal_characters(name, replacement='_'):
    # replace any potential illegal characters with 'replacement'
    for c in ['-', '[' ,']' ,'{', '}', "'", ',', ':', ' ']:
        name = name.replace(c,replacement)
    return name


