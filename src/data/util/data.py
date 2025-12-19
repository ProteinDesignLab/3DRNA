import Bio.PDB
import Bio.PDB.vectors

import torch

import numpy as np
import os
import re
import math
from typing import List, Tuple, Union
from itertools import repeat

import common.atoms
from torch.utils.data import Dataset
import seq_des.util.clean_pdb as clean_pdb
from Bio.PDB import *
import pandas as pd

N_CHI_BINS = 36
CHI_BINS = np.linspace(-180,180, num=N_CHI_BINS+1)[1:-1] # remove the edges for digitize since it uses the b.c. as bins
num_resis = len(common.atoms.label_res_rna_dict) # amino acid label res (20 amino acids)
num_atoms = len(common.atoms.atoms) # 9 atom types (N,C,O,S,P,Alkali, Earth Alkali, Transition, Other)
num_bb = len(common.atoms.bb_elem)

def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0, initial=100)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0, initial=-100).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def quantize(coords, voxel_size: Union[float, Tuple[float, ...]] = 1, *, return_index: bool = False,
             return_inverse: bool = False) -> List[np.ndarray]:
    
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = coords.squeeze()
    if torch.is_tensor(coords):
        coords = coords.cpu().detach().numpy()
    
    coords = np.delete(coords, np.where(np.any(coords > 10, axis=0)), axis=0)
    coords = np.delete(coords, np.where(np.any(coords < -10, axis=0))[0], axis=0)

    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords), return_index=True, return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    
    return outputs[0] if len(outputs) == 1 else outputs

def get_info(dataset_csv):
    """ Reads the BGSU csv file and returns the list of pdbs 
    
    return
        list of tuples (pdb_code, chain_id)
    """
        
    df = pd.read_csv(dataset_csv)
    pdb_names = list(df['pdb'])
    pdb_chains = list(df['chain'])
    pdb_list = list(tuple([pdb, chain]) for pdb, chain in zip(pdb_names, pdb_chains))
    
    '''col1 = df.iloc[:,1]
    index=0
    for rows in col1:
        ids = rows.split(",") #ids is a list
        pdb_codes = ids[0].split("|")
        pdb_list.insert(index,(pdb_codes[0], pdb_codes[2]))
        index+1'''
    return pdb_list

def download_cif(pdb, pdb_dir):
# Download cif onto directory in SCRATCH folder only if on the list. Files in this directory will then be passed in to get_dataset().

    pdb_file = os.path.join(pdb_dir, pdb + ".cif")
    pdb_clean_file = os.path.join(pdb_dir, f"{pdb}_clean.cif")
    
    if os.path.isfile(pdb_clean_file):
        return pdb_clean_file
    
    if not os.path.isfile(pdb_file):
        os.system("wget -O {} https://files.rcsb.org/download/{}.cif".format(pdb_file, pdb.upper()))

    # Skip cleaning for now
    #pdb_clean_file = clean_pdb.clean(pdb_file)

    return pdb_file #pdb_clean_file

def parser_file(pdb):
    if pdb[-3:] == "pdb":
        parser = Bio.PDB.PDBParser(QUIET=True)
    elif pdb[-3:] == "cif":
        parser = Bio.PDB.MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unknown file extension {pdb[:-3]}")
    
    structure = parser.get_structure("obj", pdb)
    return structure

def download_once(pdb_dir, csv):
    id_lst = get_info(csv)
    for pdb in id_lst:
        pdb_code = pdb[0]
        pdb_file = pdb_dir + "/" + pdb_code + ".cif"
        os.system("wget -P {} https://files.rcsb.org/download/{}.cif".format(pdb_dir, pdb_code.upper()))


def map_to_bins(chi):
    # map rotamer angles to discretized bins
    # CHI_BINS is 35 parts to makes bins of 34 + 2 intervals (2 for beyond the b.c.)
    binned_pwd = np.digitize(chi, CHI_BINS)
    return binned_pwd


def get_pdb_chains(pdb, pdb_dir, download=0, mode='train'):

    """Function to load pdb structure via Biopython and extract all chains. 
    Uses biological assembly as default, otherwise gets default pdb.
    
    Args:
        pdb (str): pdb ID.
        data_dir (str): path to pdb directory

    Returns:
        chains (list of (chain, chain_id)): all pdb chains

    """
    
    if download:
        pdb_file = download_cif(pdb, pdb_dir)
    else:
        if pdb[-3:] not in ["pdb", "cif"]:
            pdb_file = os.path.join(pdb_dir, pdb + ".cif")
        else:
            pdb_file = pdb


    if pdb_file.endswith('.pdb'):
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb, pdb_file)
    else:
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(pdb, pdb_file)

    assert len(structure) > 0, pdb

    # for assemblies -- sometimes chains are represented as different structures
    if len(structure) > 1:
        model = structure[0]
        for i in range(len(structure)):
            for c in structure[i].get_chains():
                try:
                    model.add(c)
                except Bio.PDB.PDBExceptions.PDBConstructionException:
                    continue
    else:
        model = structure[0]

    return structure, [(c, c.id) for c in model.get_chains()]

def coord_sys(pos_N, pos_CA, pos_C): #maybe change to 04', C1', and C*''
    """Defines a local reference system based on N, CA, and C atom positions"""
    
    # Define local coordinate system
    e1 = pos_C - pos_N
    e1 /= np.linalg.norm(e1)

    # Define CB positions by rotating N atoms around CA-C axis 120 degr
    pos_N_res = pos_N - pos_CA
    axis = pos_CA - pos_C
    pos_CB = np.dot(Bio.PDB.rotaxis((120.0 / 180.0) * np.pi, Bio.PDB.vectors.Vector(axis)), pos_N_res,)
    e2 = pos_CB
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    # N-C and e2 are not perfectly perpendical to one another. We adjust e2.
    e2 = np.cross(e1, -e3)
    
    # Use e3 as z-direction
    rot_matrix = np.array([e1, e2, e3])
    
    # Use backbone direction as z-direction
    # rot_matrix = np.array([e2, e3, e1])
    
    # Use sidechain direction as z-direction
    # rot_matrix = np.array([e3, e1, e2])

    return rot_matrix

def get_chi_angles(residue):

    chi = [0.0]
    mask = [0]
    
    o4 = residue["O4'"].get_vector()
    c1 = residue["C1'"].get_vector()
    
    if(residue.resname in ['A', 'G']):
        n9 = residue["N9"].get_vector()
        c4 = residue["C4"].get_vector()
        chi[0] = Bio.PDB.vectors.calc_dihedral(o4, c1, n9, c4)
        chi[0] = math.degrees(chi[0])
    else:
        n1 = residue["N1"].get_vector()
        c2 = residue["C2"].get_vector()
        chi[0] = Bio.PDB.vectors.calc_dihedral(o4, c1, n1, c2)
        chi[0] = math.degrees(chi[0])

    return np.concatenate([chi, mask])

def get_pdb_data(pdb, pdb_dir='', chain=None, allowed_resis=None, bb_only=False, voxel_size=0.5, download=1, return_gt=False,mode='train'):

    """Function to get atom coordinates and atom/residue metadata from pdb structures. 
    
    Args:
        pdb (str): pdb ID
        pdb_dir (str): path to pdb directory

    Returns:
        atom_coords (np.array): num_atoms x 3 coordinates of all retained atoms in structure
        atom_data (np.array): num_atoms x 4 data for atoms  -- [residue idx, BB ind, atom type, res type]
        res_label (np.array): num_res x 1 residue type labels (amino acid type) for all residues (to be included in training)
        chis (np.array): num_atoms x 1: chi angle

    """
    # get pdb chain data
    chains = chain.split('-')[0]
    pdb_chains = []
    temp_chains = []
    if mode == 'train':
        structure, pdb_chains = get_pdb_chains(pdb, pdb_dir, download=download, mode=mode)
        for pdb_chain, chain_id in pdb_chains:
            if chain_id in chains:
                temp_chains.append((pdb_chain, chain_id))
        pdb_chains = temp_chains
    
    elif mode == 'run':
        structure = parser_file(pdb)
        if len(structure) > 1:
            model = structure[0]
            for i in range(len(structure)):
                for c in structure[i].get_chains():
                    try:

                        model.add(c)
                    except Bio.PDB.PDBExceptions.PDBConstructionException:
                        continue
        else:
            model = structure[0]

        for c in model.get_chains():
            if c.id == chains:
                pdb_chains = [(c, c.id)]
                continue
        
        if len(pdb_chains) == 0: pdb_chains = [(c, c.id) for c in model.get_chains()]

    output_coor, output_data, res_label, chis, res_in_block = [], [], [], [], []
    res_ids = []
    atom_list = Selection.unfold_entities(structure, 'A')
    res_seq = {}
    
    allowed = False
    if allowed_resis is None:
        allowed = True
    
    ns = NeighborSearch(atom_list)
    
    # iterate over residues
    for pdb_chain, chain_id in pdb_chains:
        res_seq[chain_id] = []
        c = 0
        for residue in pdb_chain:
            res_name = residue.get_resname()
            _, res_id, _ = residue.id
            res_atoms = [atom for atom in residue.get_atoms()]
            res_atoms_coor = [list(atom.get_coord()) for atom in res_atoms]
            
            #  Creates a dictionary mapping atom_name -> atom_coord
            atom_id_list = []
            atom_coord_list = []
            for atom in res_atoms:
                atom_id = atom.get_id().strip()
                atom_coord = atom.get_coord()
                atom_id_list.append(atom_id)
                atom_coord_list.append(atom_coord)
                
            res_atoms_coord_d = dict(zip(atom_id_list, atom_coord_list))

            if allowed_resis is not None and res_id in allowed_resis:
                allowed = True

            # check if residue is a common amino acid (maybe chack later for number of atoms)
            if res_name in common.atoms.rna and allowed:
                # Check if backbone atoms are present in the RNA residue
                
                if "C1'" not in res_atoms_coord_d or "C2'" not in res_atoms_coord_d or "O4'" not in res_atoms_coord_d:
                    continue

                res_seq[chain_id].append(res_id)
                res_type_idx = common.atoms.res_label_rna_dict[res_name]
                
                # get all atoms within an 18.0 radius sphere
                atoms = ns.search(res_atoms_coor[1], 18.0, level='A')

                # generate temporary data structures
                atom_coor = np.zeros((len(atoms)*64, 3))
                atom_data = np.zeros((len(atoms)*64, 4)).astype(np.int64)
                
                rot_matrix = coord_sys(res_atoms_coord_d["O4'"], res_atoms_coord_d["C1'"], res_atoms_coord_d["C2'"])

                a = 0
                if return_gt:
                    res_in_block.append(residue["C1'"].get_coord())
                

                for atom in atoms:
                    ae = atom.element
                    bb_id = 0
                    # keep track of relevant atoms
                    if ae in common.atoms.skip_atoms:
                        continue
                    if ae in common.atoms.atoms or ae in common.atoms.all_metals:
                        elem_name = ae
                        if atom.name in common.atoms.bb_rna_atoms and ae in common.atoms.bb_elem:
                            bb_id = 1
                    else:
                        elem_name = "other"
                    if bb_only and not bb_id:
                        continue
                    ac = list(atom.get_coord())

                    if ac in res_atoms_coor:    
                        continue    
                    ac = np.array(ac) - res_atoms_coord_d["C1'"] # shift to canonicaize
                    ac = np.dot(ac, rot_matrix)     # rotate to canonicalize
                    if (ac<-10).any() or (ac>10).any():
                        continue
                    atom_resname = atom.get_parent().get_resname().replace(' ', '')

                    if atom_resname in common.atoms.rna:
                        atom_res_idx = common.atoms.res_label_rna_dict[atom_resname] #changed here
                        atom_type = common.atoms.atoms.index(elem_name)
                        atom_base = 0
                    else:
                        continue
                    #elif atom_resname in common.atoms.all_metals:
                    #    atom_res_idx = common.atoms.metals_dict[atom_resname]
                    #    atom_type = common.atoms.atoms.index(common.atoms.metal_types[atom_res_idx])
                    #    atom_base = common.atoms.metals_base[atom_resname]
                    #else:
                    #    atom_res_idx = 7 # other atom that isn't skipped, residue, or pre-defined metal
                    #    if elem_name in common.atoms.all_metals:
                    #        elem_res_idx = common.atoms.metals_dict[elem_name]
                    #        atom_type = common.atoms.atoms.index(common.atoms.metal_types[elem_res_idx])
                    #        atom_base = common.atoms.metals_base[atom_resname]
                    #    else:
                    #        atom_type = common.atoms.atoms.index(elem_name)
                    #        atom_base = 0
                                
                    atom_coor[a] = ac
                    atom_data[a] = [atom_type, bb_id, atom_res_idx, atom_base]
                    a = a + 1
                
                    radii = common.atoms.est_atomic_radii[atom_type]
                    num_voxels = radii / voxel_size

                    for v in range(int(num_voxels)):
                        val = float(v+1)/float(num_voxels)

                        atom_coor[a] = np.add(ac, [voxel_size*val, voxel_size*val, voxel_size*val])
                        atom_coor[a+1] = np.add(ac, [-voxel_size*val, voxel_size*val, voxel_size*val])
                        atom_coor[a+2] = np.add(ac, [voxel_size*val, -voxel_size*val, voxel_size*val])
                        atom_coor[a+3] = np.add(ac, [voxel_size*val, voxel_size*val, -voxel_size*val])
                        atom_coor[a+4] = np.add(ac, [-voxel_size*val, -voxel_size*val, voxel_size*val])
                        atom_coor[a+5] = np.add(ac, [voxel_size*val, -voxel_size*val, -voxel_size*val])
                        atom_coor[a+6] = np.add(ac, [-voxel_size*val, voxel_size*val, -voxel_size*val])
                        atom_coor[a+7] = np.add(ac, [-voxel_size*val, -voxel_size*val, -voxel_size*val])

                        for i in range(8):
                            atom_data[a+i] = [atom_type, bb_id, atom_res_idx, atom_base]
                        a = a + 8
                
                try:
                    if mode == 'train':
                        chis.append(get_chi_angles(residue))

                    atom_data = atom_data[:a, :]   
                    atom_coor = atom_coor[:a, :]
                    
                    output_coor.append(atom_coor)
                    output_data.append(atom_data)
                
                    res_label.append(res_type_idx)
                    res_ids.append((res_id,chain_id))

                except Exception as e:
                    print(f"Skipping {residue}. {e}")
                    print(res_atoms_coord_d.keys())
                    continue
    if mode == 'run':
        return output_coor, output_data, np.array(res_label), res_seq, res_in_block
    else:
        return res_ids, output_coor, output_data, torch.tensor(np.array(res_label)), torch.tensor(np.array(chis)) #np.array(res_label), np.array(chis)

def get_domain(domain_split):
    # function to parse CATH domain info from txt -- returns chain and domain residue IDs
    chain = domain_split[-1]

    domains = domain_split.split(",")
    domains = [d[: d.rfind(":")] for d in domains]

    domains = [(d[: d.rfind("-")], d[d.rfind("-") + 1 :]) for d in domains]
    domains = [(int(re.findall("\D*\d+", ds)[0]), int(re.findall("\D*\d+", de)[0])) for ds, de in domains]

    return chain, np.array(domains)

class PDBDataset(Dataset):
    def __init__(self, coords_dir=".", voxel_size=0.5, bb_only=False, minimal_channels=False, noise=0, set_len=0, device='cuda'):
        self.coords_dir = coords_dir
        #self.len = set_len
        self.len = len(os.listdir(coords_dir))
        self.files = os.listdir(coords_dir)
        self.cached_pt = -1
        self.voxel_size = voxel_size
        self.bb_only = bb_only
        self.noise = noise
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            #index = np.random.randint(self.len)
            #while not os.path.exists(f'{self.coords_dir}/data_{index}.pt'):
            #    index = np.random.randint(self.len)

            #output_coor, output_data, res_label, chis = torch.load(f'{self.coords_dir}/data_{index}.pt')

            pdb, chain, resi_id,  output_coor, output_data, res_label, chis = torch.load(f'{self.coords_dir}/{self.files[idx]}')
            resi_id = resi_id[0]

            if self.noise > 0:
                random_noise = np.random.normal(0, self.noise, size=np.array(output_coor).shape)
                output_coor += random_noise

            # voxelize, quantize and save coordinates and features
            coords, indices = quantize(output_coor, self.voxel_size, return_index=True)

            n = int(20/self.voxel_size)

            if self.bb_only:
                output_atom = torch.zeros((num_atoms, n + 2, n + 2, n + 2))
                output_base = torch.zeros((4, n + 2, n + 2, n + 2))
                
                indices_bb = np.nonzero(output_data[...,1][0][indices])[0]
                atom_types = output_data[...,0][0][indices]

                output_atom[atom_types[indices_bb], coords[...,0][indices_bb], coords[...,1][indices_bb], coords[...,2][indices_bb]]
                output_base[output_data[...,3][0][indices], coords[...,0], coords[...,1], coords[...,2]] = 1 
                
                ohe_data = torch.cat([output_atom[:num_bb], output_base[1:]])[:, 1:-1, 1:-1, 1:-1] 
            else:
                output_atom = torch.zeros((num_atoms, n + 2, n + 2, n + 2))
                output_bb = torch.zeros((2, n + 2, n + 2, n + 2))

                if len(output_data.shape) == 3:
                    output_data = output_data.squeeze()

                output_atom[output_data[...,0][indices], coords[...,0], coords[...,1], coords[...,2]] = 1
                output_bb[output_data[...,1][indices], coords[...,0], coords[...,1], coords[...,2]] = 1
                
                ohe_data = torch.cat([output_atom, output_bb[1].unsqueeze(0)])[:, 1:-1, 1:-1, 1:-1]

            # map chi angles to bins
            try:
                chi_binned = map_to_bins(chis)
                chi_binned[chis == 0] = 0
            except:
                return None
            if torch.isnan(chis).any() or np.isnan(chi_binned).any() or torch.isnan(ohe_data).any():
                return None
            #'pdb':pdb[0], 'chain':chain[0],'res_id': resi_id.item(), 
            return {'pdb':pdb[0], 'chain':chain[0],'res_id': resi_id.item(), 'input': ohe_data,'res_labels': np.array(res_label), 'chi_angles': chis, 'chi_binned': chi_binned}  
        except:
            return None


class get_dataset(Dataset):
    def __init__(self, input_data, pdb_dir, bb_only=False):
        """ 
        input_data: a text file containing the pdb code + chains
        pdb_dir: directory where pdbs are saved
        """
        self.pdb_dir = pdb_dir
        self.domains = get_info(input_data) # list of tuples where tuple = (pdb_code, chain_id)
        self.bb_only = bb_only

    def __len__(self):

        return len(self.domains)

    def __getitem__(self, index):
        pdb_id, chain = self.domains[index]
        return self.get_data(pdb_id, chain)

    def get_data(self, pdb, chain, directory=None):
        if directory is None:
            directory = self.pdb_dir
        try:
            return pdb, chain, get_pdb_data(pdb, directory, chain=chain, bb_only=self.bb_only)
        except Exception as e:
            print(e)
            return []
