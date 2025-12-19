import torch
import shutil
import os
from Bio.PDB import *
import wandb
import numpy as np

from torch.distributions.categorical import Categorical

import common
import seq_des.util.data as pdb_data
import seq_des.util.sampler_util as sampler_util
import seq_des.util.pyrosetta_util as pyrosetta_util

n = 40
num_resis = 4
num_atoms = len(common.atoms.atoms)
num_bb = len(common.atoms.bb_elem)

class Sampler(object):
    def __init__(self, args, models, device="cpu"):
        super(Sampler, self).__init__()
        self.pdb = args.pdb
        self.chain = args.chain

        idx = self.pdb.rfind("/")
        self.pdb_name = self.pdb[idx+1:]
        self.pdb_dir = self.pdb[:idx]
        self.log_dir = args.log_dir

        if self.pdb[-3:] == 'cif':
            self.pdb_ref_name = self.pdb_name.strip('.cif')
        elif self.pdb[-3:] == 'pdb':
            self.pdb_ref_name = self.pdb_name.strip('.pdb')
        
        self.pdb_temp = os.path.join(self.log_dir, f'{self.pdb_ref_name}_temp.pdb')
        self.pdb_curr = os.path.join(self.log_dir, f'{self.pdb_ref_name}_curr.pdb')
        self.pdb_start = os.path.join(self.log_dir, f'{self.pdb_ref_name}_start.pdb')
        self.pdb_end = os.path.join(self.log_dir, f'{self.pdb_ref_name}_end.pdb')

        self.models = models
        self.device = device
        self.seed = args.seed

        self.threshold = args.threshold
        self.randomize = args.randomize
        self.uracil = args.uracil
        self.options = np.arange(0, 4)

        self.anneal = args.anneal
        self.step_rate = args.step_rate
        self.anneal_start_temp = args.anneal_start_temp
        self.anneal_final_temp = args.anneal_final_temp
        self.accept_prob = 1
        self.iteration = 0
        self.single_res = args.single_res
        self.fixed_idx = []
        self.var_idx = []
       
        if args.fixed_idx != "":
            self.fixed_idx = sampler_util.get_idx(args.fixed_idx)
        if args.var_idx != "":
            self.var_idx = sampler_util.get_idx(args.var_idx)

        assert not (
            (len(self.fixed_idx) > 0) and (len(self.var_idx) > 0)
             ), "cannot specify both fixed and variable indices"

        self.prob_threshold = args.prob_threshold
        self.noise = args.design_noise
        self.voxel_size = args.voxel_size
        self.rosetta_energy = -1000000
        self.best_log_p = 1000000
        self.eterna_energy = 0

    def collate_data(self, atom_coords, atom_data):
        od_list = []

        for coor, data in zip(atom_coords, atom_data):
            coords, indices = pdb_data.quantize(coor, self.voxel_size, return_index=True)

            output_atom = torch.zeros((num_atoms, n + 2, n + 2, n + 2))
            output_bb = torch.zeros((2, n + 2, n + 2, n + 2))

            output_atom[data[...,0][indices], coords[...,0], coords[...,1], coords[...,2]] = 1
            output_bb[data[...,1][indices], coords[...,0], coords[...,1], coords[...,2]] = 1
            
            ohe_data = torch.cat([output_atom, output_bb[1].unsqueeze(0)])[:, 1:-1, 1:-1, 1:-1]
            od_list.append(ohe_data)

        return torch.stack(od_list, dim=0)

    def init(self):
        
        self.identifier_dict = pyrosetta_util.remove_residue(self.pdb, self.pdb_curr, chain_input=self.chain, min_atom_count=9)
        self.pose = pyrosetta_util.get_pose(self.pdb_curr)
        
        seq_start, self.skip = sampler_util.get_sequence(self.pdb_curr, chain_input=self.chain)

        for idx in self.skip[::-1]:
            seq_start.pop(idx)
            self.identifier_dict[self.chain].pop(idx)

        self.seq_start = ''.join(seq_start)
        for idx, base in enumerate(self.seq_start):
            if base not in common.atoms.rna and base != '/':
                print(f"Warning. Turning {base} at chain {self.chain} {idx+1} to U.")
                pyrosetta_util.mutate_base(self.pose, self.identifier_dict[self.chain][idx], 'u')
        
        self.pose.dump_file(self.pdb_start)

        # Rosetta score energy
        #self.gt_rosetta_energy, self.gt_rosetta_energy_min = pyrosetta_util.rna_minimize(self.pdb, self.pdb_curr)

        self.gt_atom_coords, self.gt_atom_data, self.gt_res_label, self.gt_seq, self.gt_blocks = pdb_data.get_pdb_data(self.pdb_curr, mode='run', chain=self.chain, return_gt=True)
        self.gt_ohe_data = self.collate_data(self.gt_atom_coords, self.gt_atom_data)
        
        self.n = len(self.gt_res_label)
        self.gt_log_p_per_res, self.gt_log_p_mean, self.gt_logits, self.gt_chi_logits = sampler_util.get_energy(self.models, self.gt_ohe_data, self.gt_res_label, device=self.device)

        torch.save(self.gt_log_p_per_res, os.path.join(self.log_dir, f'{self.pdb_ref_name}_s{self.seed}_gt_log_per_res.pt'))
        torch.save(self.gt_logits, os.path.join(self.log_dir, f'{self.pdb_ref_name}_s{self.seed}_gt_logits.pt'))

        self.get_blocks(single_res=self.single_res)

    def init_seq(self):

        if self.uracil:
            new_seq = np.repeat(common.atoms.res_label_rna_dict['U'], self.n)
        elif self.randomize:    # Randomize sequence, adhere to flags
            new_seq = np.random.choice(self.options, size=self.n)
        else:
            new_seq = self.gt_res_label # use gt seq as starting sequence
        
        if len(self.fixed_idx) != 0:
            new_seq[self.fixed_idx] = self.gt_res_label[self.fixed_idx]

        self.curr_seq = sampler_util.convert_seq_to_str(new_seq)

        self.pose = pyrosetta_util.rna_thread(self.pose, self.seq_start, self.curr_seq, self.pdb_start, self.identifier_dict[self.chain])
        
        self.atom_coords, self.atom_data, self.res_label, _, _ = pdb_data.get_pdb_data(self.pdb_start, mode='run', chain=self.chain)
        self.ohe_data = self.collate_data(self.atom_coords, self.atom_data)
        self.log_p_per_res, self.log_p_mean, self.logits, self.chi_logits = sampler_util.get_energy(self.models, self.ohe_data, self.res_label, device=self.device)

        shutil.copyfile(self.pdb_start, self.pdb_curr)

    def get_blocks(self, single_res=False):
        
        # get node blocks for blocked sampling
        D = sampler_util.get_C1_distance(self.gt_blocks)
        
        if single_res:  # no blocked gibbs -- sampling one res at a time
            self.blocks = [[i] for i in np.arange(D.shape[0]) if i not in self.fixed_idx]
            self.n_blocks = len(self.blocks)
        
        else:
            A = sampler_util.get_graph_from_D(D, self.threshold)
            
            self.graph = {i: np.where(A[i, :] == 1)[0] for i in range(A.shape[0])}
            nodes = np.arange(self.n)
            np.random.shuffle(nodes)
            nodes = [n for n in nodes if n not in self.fixed_idx]
            self.colors = sampler_util.color_nodes(self.graph, nodes)
            self.n_blocks = 0

            if self.colors: # check if there are any colored nodes to get n-blocks (might be empty if running NATRO on all residues in resfile)
                self.n_blocks = sorted(list(set(self.colors.values())))[-1] + 1
            self.blocks = {}

            for k in self.colors.keys():
                if self.colors[k] not in self.blocks.keys():
                    self.blocks[self.colors[k]] = []
                self.blocks[self.colors[k]].append(k)

        self.reset_block_rate = self.n_blocks

    def enforce_constraints(self, logits, idx):
        #if self.resfile:
        #    logits = self.enforce_resfile(logits, idx)

        for i in idx:
            if len(self.fixed_idx) != 0 and i in self.fixed_idx: 
                logits[i, self.gt_res_label[i]] = 100000
        return logits

    def sample(self, logits, chi_logits, idx):
        # sample residue from model conditional prob distribution at idx with current logits
        logits = self.enforce_constraints(logits, idx)
        dist = Categorical(logits=logits[idx])
        res_idx = dist.sample().cpu().data.numpy()

        assert len(res_idx) == len(idx), (len(idx), len(res_idx))

        res = []
        chi = []
        for k in list(res_idx):
            res.append(common.atoms.label_res_rna_dict[k])
            chi.append(pdb_data.CHI_BINS[torch.argmax(chi_logits[k],0).item()])

        return res, idx, chi

    def sim_anneal_step(self, e, e_old):
        delta_e = e - e_old  
        if delta_e < 0:
            accept_prob = 1.0
        else:
            if self.anneal_start_temp == 0:
                accept_prob = 0.0 
            else:
                accept_prob = torch.exp(-(delta_e) / self.anneal_start_temp).item()
        return accept_prob

    def step_T(self):
        # anneal temperature
        self.anneal_start_temp = max(self.anneal_start_temp * self.step_rate, self.anneal_final_temp)

    def step(self, step_id=1):
        # no blocks to sample (NATRO for all residues)
        if self.n_blocks == 0:
            self.step_anneal()
            return

        # random idx selection, draw sample
        idx = self.blocks[np.random.choice(self.n_blocks)]
        res, idxs, chis = self.sample(self.logits, self.chi_logits, idx)

        # mutate center residue
        pose_temp = pyrosetta_util.mutate_list(self.pose, idxs, res, chis, fixed_idx=self.fixed_idx, ident_list=self.identifier_dict[self.chain])

        self.save_pdb(pose_temp, self.pdb_temp)
        temp_coords, temp_data, temp_res_label, _, _ = pdb_data.get_pdb_data(self.pdb_temp, self.log_dir, mode='run', chain=self.chain)
        temp_ohe_data = self.collate_data(temp_coords, temp_data)

        log_p_per_res_temp, log_p_mean_temp, logits_temp, chi_logits_temp = sampler_util.get_energy(self.models, temp_ohe_data, temp_res_label, device=self.device)

        if self.anneal:
            # simulated annealing accept/reject step
            self.accept_prob = self.sim_anneal_step(log_p_mean_temp, self.log_p_mean)
            r = np.random.uniform(0, 1)
        else:
            # vanilla sampling step
            self.accept_prob = 1.0
            r = 0
        if r <= self.accept_prob:
            #self.save_pdb(pose_temp, f'{self.pdb_ref_name}_step{step_id}_pose.pdb')
            self.save_pdb(pose_temp, self.pdb_curr)
            
            self.pose = pose_temp
            self.log_p_mean = log_p_mean_temp
            self.log_p_per_res = log_p_per_res_temp
            self.logits = logits_temp
            self.chi_logits = chi_logits_temp
            self.res_label = temp_res_label

            if self.best_log_p >= self.log_p_mean.item():
                self.save_pdb(pose_temp, f'{self.pdb_ref_name}_s{self.seed}_best_logp.pdb')
                self.best_log_p = self.log_p_mean.item()
                torch.save(self.log_p_per_res, os.path.join(self.log_dir, f'{self.pdb_ref_name}_best_log_per_res.pt'))
                torch.save(self.logits, os.path.join(self.log_dir, f'{self.pdb_ref_name}_best_logits.pt'))

            wandb.log({ 'step': step_id,
                        'log_p': self.log_p_mean.item(),
                        'best_log_p': self.best_log_p})

            self.step_anneal()

    def step_anneal(self):
        # ending for step()
        if self.anneal:
            self.step_T()
        
        self.iteration += 1
        
        # reset blocks
        if self.reset_block_rate != 0 and (self.iteration % self.reset_block_rate == 0):
            self.get_blocks()

    def save_pdb(self, pose, filename):
        output_file = os.path.join(self.log_dir, filename)
        pose.dump_pdb(output_file)