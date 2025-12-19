import string

letters = string.ascii_uppercase

rename_chains = {i: letters[i] for i in range(26)}  # NOTE -- expect errors if you have more than 26 structures

all_chis = ['chi_1']

skip_res_list = [
    "HOH",
    "GOL",
    "EDO",
    "SO4",
    "EDO",
    "NAG",
    "PO4",
    "ACT",
    "PEG",
    "MAN",
    "BMA",
    "DMS",
    "MPD",
    "MES",
    "PG4",
    "TRS",
    "FMT",
    "PGE",
    "EPE",
    "NO3",
    "UNX",
    "UNL",
    "UNK",
    "IPA",
    "IMD",
    "GLC",
    "MLI",
    "1PE",
    "NO3",
    "SCN",
    "P6G",
    "OXY",
    "EOH",
    "NH4",
    "DTT",
    "BEN",
    "BCT",
    "FUL",
    "AZI",
    "DOD",
    "OH",
    "CYN",
    "NO",
    "NO2",
    "SO3",
    "H2S",
    "MOH",
    "URE",
    "CO2",
    "2NO",
]  # top ions, sugars, small molecules with N/C/O/S/P that appear to be crystal artifacts or common surface bound co-factors -- to ignore

skip_atoms = ["H", "D"]

atoms = ["N", "C", "O", "S", "P"] #, "Alkali", 'Earth Alkali', 'Transition', "other"]
bb_atoms = ["N", "C", "CA", "O"]
bb_elem = ["N", "C", "O"]

bb_rna_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'", "P", "OP1", "OP2"]
bb_rna_elem = ["N", "C", "O", "P"]

alkali = ['LI', 'NA', 'K', 'RB']
earth_alkali = ['MG', 'CA']
transition = ['MN', 'FE', 'FE2', 'CO', 'CU', 'CU1', 'ZN', 'NI']

all_metals = ['LI', 'NA', 'K', 'RB', 'MG', 'CA', 'MN', 'FE', 'FE2', 'CO', 'CU', 'CU1', 'ZN', 'NI']

                # "N", "C", "O", "S", "P", "Alkali", 'Earth Alkali', 'Transition', "other"
est_atomic_radii = [0.5, 0.5, 0.5, 1, 1, 2.5, 2, 1.5, 0.5]

metal_types = {
    4: 'Alkali',
    5: 'Earth Alkali',
    6: 'Transition'
}
metals_dict = {
    'LI': 4,
    'NA': 4,
    'K': 4,
    'RB': 4,
    'MG': 5,
    'CA': 5, 
    'CU': 6,
    'CU1': 6,
    'ZN': 6,
    'CO': 6,
    'NI': 6,
    'MN': 6,
    'FE': 6,
    'FE2': 6
}
metals_base = {
    'LI': 1,
    'NA': 1,
    'K': 1,
    'RB':1,
    'MG': 1,
    'CA': 1, 
    'CU': 2,
    'CU1': 3,
    'ZN': 2,
    'CO': 2,
    'NI': 2,
    'MN': 1,
    'FE': 1,
    'FE2': 2
}
# maybe add psuedouridine later
rna = [
    "A",
    "C",
    "G",
    "U",
]
res_label_rna_dict = {
    "A":0,
    "C":1,
    "G":2,
    "U":3,
}

label_res_rna_dict = {
    0:"A",
    1:"C",
    2:"G",
    3:"U",
}


freqs = {
    'A': 0.24232641428535398,
    'C': 0.24597930170580173,
    'G': 0.31713073890051874,
    'U': 0.19456354510832555
}

res_weights = [1.07583418, 1.11530398, 0.89713322, 0.94409938]

chi_weights = [1.82675666e-01, 8.62971139e-02, 1.42847476e-01, 4.82624600e-01,
                6.23817967e-01, 7.71564327e-01, 9.38222222e-01, 1.24763593e+00,
                1.83246528e+00, 2.79232804e+00, 3.35079365e+00, 3.66493056e+00,
                3.90925926e+00, 6.17251462e+00, 2.34555556e+01, 2.93194444e+01,
                1.17277778e+02, 5.86388889e+01, 2.93194444e+01, 2.93194444e+01,
                1.46597222e+01, 1.46597222e+01, 6.17251462e+00, 5.33080808e+00,
                4.18849206e+00, 9.02136752e+00, 2.34555556e+01, 1.95462963e+01,
                2.34555556e+01, 5.86388889e+01, 2.93194444e+01, 3.90925926e+01,
                2.93194444e+01, 1.17277778e+01, 3.25771605e+00, 6.40862174e-01]

# resfile commands where values are amino acids allowed by that command 
resfile_commands = {
    "ALLAA": {'H', 'K', 'R', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G', 'C'},
    "ALLAAwc": {'H', 'K', 'R', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G', 'C'},
    "ALLAAxc": {'H', 'K', 'R', 'D', 'E', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G'},
    "POLAR": {'E', 'H', 'K', 'N', 'R', 'Q', 'D', 'S', 'T'},
    "APOLAR": {'P', 'M', 'Y', 'V', 'F', 'L', 'I', 'A', 'C', 'W', 'G'},
}
