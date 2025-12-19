import Bio.PDB
import sys
from seq_des.util.clean_pdb import clean
import seq_des.util.sampler_util as sampler_util
import common.atoms

# For importing pyrosetta in python 3.7
try:
    sys.path.remove("/home/groups/possu/pyrosetta/PyRosetta4.MinSizeRel.python38.linux.release-341")
    sys.path.append("/home/groups/possu/pyrosetta/PyRosetta4.MinSizeRel.python39.linux.release-335")
except Exception as e:
    print("Already altered path for pyrosetta")
    print(e)
import pyrosetta
import pyrosetta.rosetta as rosetta
import pyrosetta.rosetta.core as core


pyrosetta.init("-ex1 -ignore_zero_occupancy false -no_optH false")
ROSETTA = "/home/groups/possu/working_build/200522_master/Rosetta/main/source/bin/rna_thread.default.linuxgccrelease"

scorefxn = core.scoring.ScoreFunctionFactory.create_score_function("stepwise/rna/rna_res_level_energy7beta.wts")

def get_io(pdb_output):
    if pdb_output[-3:] == 'cif':
        return BIO.PDB.MMCIFIO()
    elif pdb_output[-3:] == 'pdb':
        return Bio.PDB.PDBIO()

def remove_residue(pdb_input, pdb_output, chain_input=None, exclude_resis=[], min_atom_count=9):
    structure = sampler_util.parser_file(pdb_input)[0]
    
    identifier_dict = {}
    remove_res_list = []
    counter = 1
    for chain in structure:
        identifier_dict[chain.id] = []
        for res in chain:
            resname = res.get_resname().strip()
            atom_count = len([atom for atom in res])
            if (chain_input is None or chain.id == chain_input) and resname in common.atoms.rna and (resname in exclude_resis or atom_count < min_atom_count):
                remove_res_list.append(res)
            else:
                identifier_dict[chain.id].append(counter)
                counter += 1
    for res in remove_res_list:
        chain.detach_child(res.id)
        res.detach_parent()

    io = get_io(pdb_output)
    io.set_structure(structure)
    io.save(pdb_output)
    
    return identifier_dict

def remove_base(pdb_input, pdb_output, residx_list):
    structure = sampler_util.parser_file(pdb_input)[0]
    bb_atom_list = ["C1'", "C2'", "C3'", "C4'", "C5'", "P"]
    bb_atom_list += ["O2'", "O3'", "O4'", "O5'"]
    bb_atom_list += ["OP1", "OP2", "OP3", "O1A", "O2A", "O3A"]
    cur_residx = 0
    for chain in structure:
        for res in chain:
            if cur_residx in residx_list:
                remove_atom_list = []
                for atom in res:
                    atom_name = atom.get_name().strip()
                    if atom_name not in bb_atom_list:
                        remove_atom_list.append(atom)
                for atom in remove_atom_list:
                    atom.detach_parent()
                    res.detach_child(atom.id)
                res.id = (' ', res.id[1], ' ')
                res.resname = "N"
            cur_residx += 1

    io = get_io(pdb_output)
    io.set_structure(structure)
    io.save(pdb_output)    

def mutate_base(pose, idx, base):
    core.pose.rna.mutate_position(pose, idx, base)

def get_pose(pdb_input):
    return pyrosetta.pose_from_pdb(pdb_input)

def rna_minimize(pdb_output=None, pdb_input='', pose=None, check_score=False):
    if pose == None:
        pose = get_pose(pdb_input)

    if check_score:
        scorefxn(pose)
        total_score = pose.energies().total_energies()[core.scoring.ScoreType.total_score]
    
    rna_min_options = core.import_pose.options.RNA_MinimizerOptions()
    rna_min_options.set_max_iter(1000)
    rna_min_options.set_minimizer_use_coordinate_constraints(True)
    rna_minimizer = rosetta.protocols.rna.denovo.movers.RNA_Minimizer(rna_min_options)
    rna_minimizer.apply(pose)

    if pdb_output is not None:
        pose.dump_file(pdb_output)

    if check_score:
        scorefxn(pose)
        total_score_new = pose.energies().total_energies()[core.scoring.ScoreType.total_score]
        print(total_score, total_score_new)
        return pose, total_score, total_score_new
    
    return pose


def mutate_list(pose, idx_list, res_list, chi_list, fixed_idx=[], var_idx=[], ident_list=[]):
    assert len(idx_list) == len(res_list), (len(idx_list), len(res_list))
    for idx, base, chi in zip(idx_list, res_list, chi_list):
        if len(fixed_idx) > 0 and idx in fixed_idx:
            continue
        if len(var_idx) > 0 and idx not in var_idx:
            continue
        mutate_base(pose, ident_list[idx], base.lower())
        if not rna_minimize:
            pose.set_chi(ident_list[idx], chi)
    if rna_minimize:
        return rna_minimize(pose=pose)
    return pose


def rna_thread(pose, curr_seq, new_seq, pdb_output=None, ident_list=[]):
    curr_seq = curr_seq.lower()
    new_seq = new_seq.lower()

    for idx, base in enumerate(new_seq):
        if curr_seq[idx] != base:
            mutate_base(pose, ident_list[idx], base)
    
    if pdb_output is not None:
        pose.dump_file(pdb_output)
    
    return pose
