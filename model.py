import deformation_model as m
import STL as stl
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = m.UNet()
        self.stl = stl.Dense2DSpatialTransformer()

    def forward(self, m, f):
        phi = self.m(f, m)
        mphi = self.stl(m, phi)

        return mphi, phi
