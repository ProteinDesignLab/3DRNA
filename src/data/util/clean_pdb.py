# Clean PDB Script
# from https://github.com/harryjubb/pdbtools/blob/master/clean_pdb.py
# ================
# 
# 
# PDB File Issues
# ---------------
# 
# - Multiple models ** DONE **
# - Multiple occupancies/alternate locations  ** DONE **
#   - Pick highest occupancy, remove alternate locations  ** DONE **
#   - Set occupancies to 1.00  ** DONE **
# - Missing atoms, residues ** NOT DEALT WITH **
# - Chain breaks ** DONE **
#   - Output to file: or CA or N or C position ** DONE **
# - Selenomets to Mets ** DONE **
# - Nonstandard res to ATOM records ** DONE **

# IMPORTS
import argparse
import logging
import operator
import os
import sys
import traceback
from functools import reduce

from collections import OrderedDict

from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import Select
from Bio.PDB.Polypeptide import PPBuilder

# CONSTANTS
PDB_LINE_TEMPLATE = '{record: <6}{serial: >5} {atom_name: ^4}{altloc: ^1}{resname: ^3} {chain_id: ^1}{resnum: >4}{icode: ^1}   {x: >8.3f}{y: >8.3f}{z: >8.3f}{occ: >6.2f}{tfac: >6.2f}          {element: >2}{charge: >2}'


#############
# CLEAN PDB #
#############

def create_MMCIF_dict(pdb_code):
    MMCIF_d = {}
    MMCIF_d["data_"] = pdb_code
    MMCIF_d["_atom_site.group_PDB"] = []
    MMCIF_d["_atom_site.id"] = []
    MMCIF_d["_atom_site.type_symbol"] = []
    MMCIF_d["_atom_site.label_atom_id"] = []
    MMCIF_d["_atom_site.label_alt_id"] = []
    MMCIF_d["_atom_site.label_comp_id"] = []
    MMCIF_d["_atom_site.label_asym_id"] = []
    MMCIF_d["_atom_site.label_entity_id"] = []
    MMCIF_d["_atom_site.label_seq_id"] = []
    MMCIF_d["_atom_site.pdbx_PDB_ins_code"] = []
    MMCIF_d["_atom_site.Cartn_x"] = []
    MMCIF_d["_atom_site.Cartn_y"] = []
    MMCIF_d["_atom_site.Cartn_z"] = []
    MMCIF_d["_atom_site.occupancy"] = []
    MMCIF_d["_atom_site.B_iso_or_equiv"] = []
    MMCIF_d["_atom_site.pdbx_formal_charge"] = []
    MMCIF_d["_atom_site.auth_seq_id"] = []
    MMCIF_d["_atom_site.auth_comp_id"] = []
    MMCIF_d["_atom_site.auth_asym_id"] = []
    MMCIF_d["_atom_site.auth_atom_id"] = []
    MMCIF_d["_atom_site.pdbx_PDB_model_num"] = []
    return MMCIF_d
    

def clean(pdb_path, remove_waters=True, keep_hydrogens = False, informative_filenames=False):
    
    pdb_noext, pdb_ext = os.path.splitext(pdb_path)
    pdb_ext = pdb_ext.replace('.', '')
    
    mode = None
    if pdb_path.endswith(".pdb"):
        pdb_parser = PDBParser()
        structure = pdb_parser.get_structure(os.path.split(os.path.splitext(pdb_path)[0])[1], pdb_path)
        mode = "pdb"
    else:
        cif_parser = MMCIFParser()
        structure = cif_parser.get_structure(os.path.split(os.path.splitext(pdb_path)[0])[1], pdb_path)
        mode = "cif"
    
    
    # OUTPUT LABEL
    output_label = 'clean'
    
    if informative_filenames:
        if remove_waters:
            output_label = output_label + '_dry'
        
        if keep_hydrogens:
            output_label = output_label + '_kh'
    
    # REMOVE MULTIPLE MODELS
    # BY TAKING THE FIRST MODEL
    model = structure[0]
    
    # RAISE AN ERROR FOR TOO MANY ATOMS
    if (mode == "pdb") and (len(list(model.get_atoms())) > 99999):
        try:
            raise ValueError('More than 99999 atoms in the PDB model!')
        except:
            traceback.print_exc(file=sys.stdout)
            exit(9)
    
    # DETERMINE POLYPEPTIDES AND CHAIN BREAKS
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(model, aa_only=False)
    
    # MAKE DATA STRUCTURES FOR CHAIN POLYPEPTIDES
    chain_ids = set([x.id for x in model.child_list])
    chain_pieces = OrderedDict()
    chain_polypeptides = OrderedDict()
    chain_break_residues = OrderedDict()
    chain_sequences = OrderedDict()
    
    for chain_id in chain_ids:
        chain_pieces[chain_id] = 0
        chain_break_residues[chain_id] = []
        chain_polypeptides[chain_id] = []
    
    # GET ALL POLYPEPTIDE RESIDUES IN THE MODEL
    polypeptide_residues = []
    
    for pp in polypeptides:
        for res in pp:
            polypeptide_residues.append(res)
    
    # GET THE CHAIN_ID(S) ASSOCIATED WITH EACH POLYPEPTIDE
    polypeptide_chain_id_sets = [set([k.get_parent().id for k in x]) for x in polypeptides]
    
    for e, polypeptide_chain_id_set in enumerate(polypeptide_chain_id_sets):
    
        # WARN IF NOT JUST ONE CHAIN ID ASSOCIATED WITH THE POLYPEPTIDE
        if len(polypeptide_chain_id_set) != 1:
            logging.warn('A polypeptide had {} chains associated with it: {}'.format(len(polypeptide_chain_id_set),
                                                                                   polypeptide_chain_id_set))
    
        for polypeptide_chain_id in polypeptide_chain_id_set:
            chain_pieces[polypeptide_chain_id] = chain_pieces[polypeptide_chain_id] + 1
    
            # ADD FIRST AND LAST RESIDUE TO THE CHAIN BREAK RESIDUES (POLYPEPTIDE TERMINAL RESIDUES)
            chain_break_residues[polypeptide_chain_id] = chain_break_residues[polypeptide_chain_id] + [polypeptides[e][0], polypeptides[e][-1]]
            chain_polypeptides[polypeptide_chain_id] = chain_polypeptides[polypeptide_chain_id] + [polypeptides[e]]
            
    # POP OUT THE FIRST AND LAST RESIDUES FROM THE CHAIN BREAK RESIDUES
    # TO REMOVE THE GENUINE TERMINI
    for chain_id in chain_break_residues:
        chain_break_residues[chain_id] = chain_break_residues[chain_id][1:-1]
    
    all_chain_break_residues = reduce(operator.add, chain_break_residues.values())
    
    # MAKE THE CHAIN SEQUENCES FROM THE CHAIN POLYPEPTIDE PIECES
    for chain_id in chain_polypeptides:
    
        pp_seqs = [str(x.get_sequence()) for x in chain_polypeptides[chain_id]]
        
        if pp_seqs:
            chain_sequences[chain_id] = reduce(operator.add, pp_seqs)
    
    # WRITE OUT CLEANED PDB
    # MANY OF THE ISSUES ARE SOLVED DURING THE WRITING OUT
    if mode == "pdb":
        fo = open(f'{pdb_noext}_{output_label}.{pdb_ext}', 'w+')
    else:
        pdb_code = os.path.basename(pdb_path)[:-4]
        MMCIF_d = create_MMCIF_dict(pdb_code)

    atom_serial = 1
    for residue in model.get_residues():

        # REMOVE WATERS IF FLAG SET
        if residue.get_full_id()[3][0] == 'W' and remove_waters:
            continue

        record = 'ATOM'

        # SET HETATM RECORD IF IT WAS ORIGINALLY A HETATM OR WATER
        if residue.get_full_id()[3][0] == 'W' or residue.get_full_id()[3][0].startswith('H_'):
            record = 'HETATM'

        # SET ATOM RECORD IF THE RESIDUE IS IN A POLYPEPETIDE
        if residue in polypeptide_residues:
            record = 'ATOM'


        for atom in residue.child_list:

            # DEAL WITH DISORDERED ATOMS
            if atom.is_disordered():
                atom = atom.disordered_get()

            # REMOVE HYDROGENS
            #if not keep_hydrogens:
            #    if atom.element.strip() == 'H':
            #        continue

            # CONVERT SELENOMETHIONINES TO METHIONINES
            if residue in polypeptide_residues and (residue.resname == 'MSE' or residue.resname == 'MET'):

                residue.resname = 'MET'

                if atom.name == 'SE' and atom.element == 'SE':
                    atom.name = 'SD'
                    atom.element = 'S'

            # FIX ATOM NAME BUG
            if len(atom.name) == 3:
                atom.name = ' ' + atom.name

            # PDB OUTPUT
            # ATOM SERIALS ARE RENUMBERED FROM 1
            # ALTLOCS ARE ALWAYS BLANK
            # CHARGES ARE ALWAYS BLANK(?)
            # OCCUPANCIES ARE ALWAYS 1.00
            if mode == "pdb":
                output_line = PDB_LINE_TEMPLATE.format(record=record,
                                                       serial=atom_serial,
                                                       atom_name=atom.name.strip(),
                                                       altloc=' ',
                                                       resname=residue.resname,
                                                       chain_id=residue.get_parent().id,
                                                       resnum=residue.get_id()[1],
                                                       icode=residue.get_id()[2],
                                                       x=float(atom.coord[0]),
                                                       y=float(atom.coord[1]),
                                                       z=float(atom.coord[2]),
                                                       occ=1.00,
                                                       tfac=atom.bfactor,
                                                       element=atom.element,
                                                       charge='')
                fo.write('{}\n'.format(output_line))
            else:
                MMCIF_d["_atom_site.group_PDB"].append(record)
                MMCIF_d["_atom_site.id"].append(str(atom_serial))
                MMCIF_d["_atom_site.type_symbol"].append(atom.element)
                MMCIF_d["_atom_site.label_atom_id"].append(atom.name.strip())
                MMCIF_d["_atom_site.label_alt_id"].append(".")
                MMCIF_d["_atom_site.label_comp_id"].append(residue.resname)
                MMCIF_d["_atom_site.label_asym_id"].append(residue.get_parent().id)
                MMCIF_d["_atom_site.label_entity_id"].append("1")
                MMCIF_d["_atom_site.label_seq_id"].append(str(residue.get_id()[1]))
                if residue.get_id()[2] == " ":
                    MMCIF_d["_atom_site.pdbx_PDB_ins_code"].append("?")
                else:
                    MMCIF_d["_atom_site.pdbx_PDB_ins_code"].append(residue.get_id()[2])
                MMCIF_d["_atom_site.Cartn_x"].append(f"{float(atom.coord[0]):.3f}") # .3f
                MMCIF_d["_atom_site.Cartn_y"].append(f"{float(atom.coord[1]):.3f}") # .3f
                MMCIF_d["_atom_site.Cartn_z"].append(f"{float(atom.coord[2]):.3f}") # .3f
                MMCIF_d["_atom_site.occupancy"].append("1.00")
                MMCIF_d["_atom_site.B_iso_or_equiv"].append(f"{atom.bfactor:.2f}") # .2f
                MMCIF_d["_atom_site.pdbx_formal_charge"].append("?")
                MMCIF_d["_atom_site.auth_seq_id"].append(str(residue.get_id()[1]))
                MMCIF_d["_atom_site.auth_comp_id"].append(residue.resname)
                MMCIF_d["_atom_site.auth_asym_id"].append(str(residue.get_parent().id))
                MMCIF_d["_atom_site.auth_atom_id"].append(atom.name.strip())
                MMCIF_d["_atom_site.pdbx_PDB_model_num"].append("1")

            atom_serial += 1

    if mode == "pdb":
        fo.close()
    else:
        io = MMCIFIO()
        io.set_dict(MMCIF_d)
        io.save(f'{pdb_noext}_{output_label}.{pdb_ext}')
    
    return f'{pdb_noext}_{output_label}.{pdb_ext}'
    