import argparse
import random
import time
import numpy as np
import torch
import os

class RunManager(object):
    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # training parameters
        self.parser.add_argument("--batchSize", type=int, default=128, help="input batch size")
        self.parser.add_argument("--cuda", type=bool, default=1, help="use cuda if available")

        self.parser.add_argument("--nf", type=int, default=32, help="base number of filters")
        self.parser.add_argument("--voxel_size", type=float, default=0.5, help="size of voxel grid")
        self.parser.add_argument("--bb_only", type=int, default=0, help="train with backbone only (1), default is with sidechains (0)")

        # chi parameters
        self.parser.add_argument("--use_chi_bin", type=bool, default=1, help="Use binning (default: False)")
        self.parser.add_argument("--weight_chi", type=float, default=1, help="Weighting for chi angle prediction")
        self.parser.add_argument("--noise", type=float, default=0, help="Noise to apply to voxel values")
        self.parser.add_argument("--single_res", type=bool, default=1, help='0 - single mutations, 1 - blocked')

        self.parser.add_argument("--fixed_idx", type=str, default="", help="Path to txt file listing pose indices that should NOT be designed/packed, all other side-chains will be designed. 0-indexed")
        self.parser.add_argument("--var_idx", type=str, default="", help="Path to txt file listing pose indices that should NOT be designed/packed, all other side-chains will be designed. 0-indexed")


        self.parser.add_argument("--epochs", type=int, default=100, help="enables cuda")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--reg", type=float, default=1e-3, help="L2 regularization")
        self.parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam. default=0.5")
        self.parser.add_argument("--momentum", type=float, default=0.01, help="momentum for batch norm")

        self.parser.add_argument("--model", type=str, default="", help="path to saved pretrained model for resuming training",)
        self.parser.add_argument("--optimizer", type=str, default="", help="path to saved optimizer params")
        self.parser.add_argument("--validation_frequency", type=int, default=50, help="how often to validate during training",)
        self.parser.add_argument("--save_frequency", type=int, default=100, help="how often to save models")
        self.parser.add_argument("--chunk_size", type=int, default=1000, help="chunk size for saved coordinate tensors")

        self.parser.add_argument("--wandb_path", type=str, default="/scratch/users/gelnesr/logs")
        self.parser.add_argument("--pdb_dir", type=str, default="/scratch/users/gelnesr/datasets/rna_dataset")
        self.parser.add_argument("--log_dir", type=str, default="/home/users/gelnesr/rna_seq_des/output", help="where the outputs will be dumped")
        self.parser.add_argument("--save_dir", type=str, default="/scratch/users/gelnesr/data/rna_train", help="where coordinates for training are stored")
        self.parser.add_argument("--input_data", type=str, default="../data/BGSUdataset_train.csv", help="default input data file for training")
        self.parser.add_argument("--coord_dir", type=str, default="/scratch/users/gelnesr/data", help="path for data coordinates")

        # design inputs
        self.parser.add_argument("--pdb", type=str, default="/home/users/gelnesr/rna_seq_des/input/5XTM_clean.cif", help="Input PDB or CIF")
        self.parser.add_argument("--init_model", type=str, default="", help="Path to baseline model (conditioned on backbone atoms only)")
        self.parser.add_argument("--model_list", "--list", nargs="+", help="Paths to conditional models",
                                 default=['/home/users/gelnesr/rna_seq_des/models/conditional_model_f1.pt'])
        self.parser.add_argument("--chain", type=str, default=None, help="which chain, default is all chains")

        # saving / logging
        self.parser.add_argument("--seed", default=7, type=int, help="Random seed. Design runs are non-deterministic.")
        self.parser.add_argument("--save_rate", type=int, default=10, help="How often to save intermediate designed structures",)

        # design parameters
        #self.parser.add_argument("--no_init_model", type=int, default=0, choices=(0, 1), help="Do not use baseline model to initialize sequence/rotmaers.",)
        self.parser.add_argument('--uracil', default=1, type=bool)
        self.parser.add_argument('--design_noise', default=0.0, type=float)
        self.parser.add_argument('--prob_threshold', default=0.75, type=float)
        
        self.parser.add_argument("--randomize", type=int, default=0, choices=(0, 1),
            help="Randomize starting sequence/rotamers for design. Toggle OFF to keep starting sequence and rotamers")
        self.parser.add_argument("--threshold", type=float, default=20,
            help="Threshold in angstroms for defining conditionally independent residues for blocked sampling (should be greater than ~17.3)")

        self.parser.add_argument("--resfile", type=str, default="", help="Specify path to a resfile to enforce constraints on particular residues")
        self.parser.add_argument("--input_index", type=int, default=0)
        # optimization / sampling parameters
        self.parser.add_argument("--anneal", type=int, default=1, choices=(0, 1), help="Option to do simulated annealing of average negative model pseudo-log-likelihood. Toggle OFF to do vanilla blocked sampling")
        self.parser.add_argument("--do_mcmc", type=int, default=0, help="Option to do Metropolis-Hastings")
        self.parser.add_argument("--step_rate", type=float, default=0.9, help="Multiplicative step rate for simulated annealing")
        self.parser.add_argument("--anneal_start_temp", type=float, default=0.01, help="Starting temperature for simulated annealing")
        self.parser.add_argument("--anneal_final_temp", type=float, default=0.0, help="Final temperature for simulated annealing")
        self.parser.add_argument("--n_iters", type=int, default=5, help="Total number of iterations")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, arg_list=None):
        if arg_list is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(args=arg_list)
            
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return self.args
