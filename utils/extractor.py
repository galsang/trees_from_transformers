import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, n_hidden):
        super(Extractor, self).__init__()
        self.linear = nn.Linear(n_hidden * 2, 1)
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.uniform_(self.linear.bias, 0)

    def forward(self, l, r):
        h = torch.cat([l, r], dim=-1)
        o = self.linear(h)
        # (seq_len-1)
        return o.squeeze(-1)

    def loss(self, d, gold):
        assert len(d) == len(gold)
        gold = d.new_tensor(gold)
        l = 0
        for i in range(len(d)):
            for j in range(i+1, len(d)):
                l += F.relu(1 - torch.sign(gold[i]- gold[j]) * (d[i] - d[j]))
        return l
