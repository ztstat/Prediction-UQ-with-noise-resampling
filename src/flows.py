import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))
    def forward(self, z):
        wu    = torch.dot(self.w, self.u)
        m     = -1 + F.softplus(wu)
        w_sq  = (self.w**2).sum()
        u_hat = self.u + ((m - wu)/(w_sq+1e-10)) * self.w
        lin   = z @ self.w.unsqueeze(1) + self.b
        return z + u_hat * torch.tanh(lin)

class PlanarFlowModel(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(num_flows)])
    def forward(self, z):
        for f in self.flows:
            z = f(z)
        return z