from __future__ import absolute_import, division, print_function
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from test_dev import TESTER
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()
import cv2
import torch
import random
import numpy as np
from PIL import Image 
seed = opts.seed
# Set the random seed for Python's random module
random.seed(seed)
# Set the random seed for Numpy
np.random.seed(seed)
# Set the random seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    tester = TESTER(opts)
    tester.test()
