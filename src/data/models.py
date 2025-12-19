import torch
import torch.nn as nn
import common.atoms

chi_bins = 36
num_resis = len(common.atoms.rna)
def init_ortho_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv1d):
            torch.nn.init.orthogonal_(module.weight)
        if isinstance(module, nn.Conv3d):
            torch.nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.orthogonal_(module.weight)

class seqPred(nn.Module):
    def __init__(self, nic, nf=32, momentum=0.01, drop=0.1, use_chi_bin=True):
        super(seqPred, self).__init__()
        self.use_chi_bin = use_chi_bin
        self.nic = nic
        self.model = nn.Sequential(
            # 40 -- 20
            nn.Conv3d(nic, nf, 9, 2, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf, nf, 7, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf, nf, 6, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            # 10 -- 5
            nn.Conv3d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            # 5 -- 1
            nn.Conv3d(nf * 2, nf * 4, 5, 1, 0, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
        )

        # res pred
        self.out = nn.Sequential(
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(nf * 4, len(common.atoms.label_res_rna_dict.keys()), 3, 1, 1, bias=False),
        )
            # chi feat vec -- condition on residue and env feature vector
        self.chi_feat = nn.Sequential(
            nn.Conv1d(nf * 4 + 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(drop),
        )
        
        # chi 1 pred -- condition on chi feat vec
        if self.use_chi_bin:
            self.chi_1_out = nn.Sequential(
                nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm1d(nf * 4, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm1d(nf * 4, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Conv1d(nf * 4, chi_bins , 3, 1, 1, bias=False),
            )
        else:
            self.chi_1_out = nn.Sequential(
                nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm1d(nf * 4, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm1d(nf * 4, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Conv1d(nf * 4, 2, 3, 1, 1, bias=False), # output 2 values for the angle
                nn.Hardtanh(min_val=-1,max_val=1)
            )

    def get_feat(self, inputs):
        bs = inputs.size()[0]
        feat = self.model(inputs).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)
        res_onehot = torch.zeros(size=(len(res_pred), num_resis), dtype=torch.int8).to(res_pred.device)
        
        chi_init = torch.cat([feat, res_onehot[..., None]], 1)
        chi_feat = self.chi_feat(chi_init)

        chi_1_pred = self.chi_1_out(chi_feat).view(bs, -1)
        return res_pred, chi_1_pred
        
    def forward(self, inputs, res_onehot):

        bs = len(res_onehot)
        feat = self.model(inputs).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)
            
        chi_init = torch.cat([feat, res_onehot[..., None]], 1)
        chi_feat = self.chi_feat(chi_init)

        # condition on true residue type and previous ground-truth rotamer angles
        chi_1_pred = self.chi_1_out(chi_feat).view(bs, -1)

        if not self.use_chi_bin: # get angle in radians
            chi_1_pred = torch.atan2(chi_1_pred[:,0], chi_1_pred[:,1])

        return res_pred, chi_1_pred
