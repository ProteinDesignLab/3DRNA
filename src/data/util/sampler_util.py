import numpy as np
import torch
import Bio.PDB
import common
import torch.nn.functional as F
from seq_des.util.data import get_pdb_data, quantize
import seq_des.util.pyrosetta_util as pyrosetta_util
N_CHI_BINS = 36
CHI_BINS = np.linspace(-180,180, num=N_CHI_BINS+1)[1:-1]

def parser_file(pdb):
    if pdb[-3:] == "pdb":
        parser = Bio.PDB.PDBParser(QUIET=True)
    elif pdb[-3:] == "cif":
        parser = Bio.PDB.MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unknown file extension {pdb[:-3]}")
    structure = parser.get_structure("obj", pdb)
    return structure

def get_graph_from_D(D, threshold):
    A = np.zeros_like(D)
    A[D < threshold] = 1
    return A

def get_idx(filename):
    idx = []
    with open(filename, "r") as f:
        for line in f:
            for field in line.split(','):
                field = field.replace('-', ' - ')
                tokens = field.split()
                if '-' in tokens:
                    start, end = int(tokens[0]), int(tokens[2])
                    idx.extend(range(start, end+1))
                else:
                    idx.append(int(tokens[0]))
    return idx

def get_sequence(pdb_input, chain_input=None):
    structure = parser_file(pdb_input)[0]
    seq = []
    skip = []
    sk = 0
    for chain in structure:
        if chain_input is None or chain.id == chain_input:
            for res in chain:
                if len(res.get_resname().strip()) == 1:
                    resname = res.get_resname().strip()
                    seq.append(resname)
                else:
                    seq.append('/')
                    skip.append(sk)
                sk+=1
    return seq, skip

def convert_seq_to_str(seq):
    return ''.join([common.atoms.label_res_rna_dict[seq[i]] for i in range(len(seq))])

def seq_similarity(seq1, seq2):
    num = 0
    den = len(seq1)
    for b1, b2 in zip(seq1, seq2):
        num += (b1 == b2)
    return num/den

def get_energy_from_logits(logits, res_idx, mask=None, baseline=0):
    # get negative log prob from logits
    log_p = -F.log_softmax(logits, -1).gather(1, res_idx[:, None])
    if mask is not None:
        log_p[mask == 1] = baseline
    log_p_mean = log_p.mean()

    return log_p, log_p_mean

def get_energy(models, atom_data, res_label, device='cpu'):

    # get residue and rotamer logits
    atom_data = atom_data.to(device)
    res_label = torch.tensor(res_label, dtype=torch.int64, device=device)
    logits = []
    chi_logits= []
    with torch.no_grad():
        for model in models:
            logit, chi_logit = model.get_feat(atom_data)
            logits.append(logit[None])
            chi_logits.append(chi_logit[None])
    
    chi_logits = torch.cat(chi_logits, 0).mean(0)
    logits = torch.cat(logits, 0).mean(0)
    
    # get model negative log probs (model energy) 
    log_p_per_res, log_p_mean = get_energy_from_logits(logits, res_label)
    return log_p_per_res, log_p_mean, logits, chi_logits

def get_C1_distance(A):
    A = np.array(A)
    D = np.sqrt(np.sum((A[:, None].repeat(len(A), axis=1) - A[None].repeat(len(A), axis=0)) ** 2, -1))
    return D

# from https://codereview.stackexchange.com/questions/203319/greedy-graph-coloring-in-python
def color_nodes(graph, nodes):
    color_map = {}
    # Consider nodes in descending degree
    for node in nodes:  # sorted(graph, key=lambda x: len(graph[x]), reverse=True):
        neighbor_colors = set(color_map.get(neigh) for neigh in graph[node])
        color_map[node] = next(color for color in range(len(graph)) if color not in neighbor_colors)
    return color_map