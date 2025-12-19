import torch

""" accuracy eval fns """

# A C G U
label_rna = {0:0, 1:1, 2:0, 3:1}
label_rna_type = {0:'purine', 1: 'pyrimidine'}

def get_acc(logits, label, label_dict=None, ignore_idx=None):
    pred = torch.argmax(logits, 1)

    if label_dict is not None:
        pred = torch.LongTensor([label_dict[p.item()] if p.item() in label_dict else 23 for p in pred])
        label = torch.LongTensor([label_dict[l.item()]if l.item() in label_dict else 23 for l in label])
    if ignore_idx is None:
        acc = float((pred == label).sum(-1)) / label.size()[0]
    else:
        # case when all data in a batch is to be ignored
        if len(label[label != ignore_idx]) == 0:
            acc = 0.0
        else:
            acc = float((pred[label != ignore_idx] == label[label != ignore_idx]).sum(-1)) / len(label[label != ignore_idx]) 

    return acc

def get_base_type_acc(logits, label):
    pred = torch.argmax(logits, 1)

    pred = torch.LongTensor([label_rna[p.item()] for p in pred])
    label = torch.LongTensor([label_rna[l.item()] for l in label])

    acc = float((pred == label).sum(-1)) / label.size()[0]
    return acc
