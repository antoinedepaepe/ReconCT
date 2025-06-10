from src.operators.sparsifier import Sparsifier
from src.operators.operator import Operator
# import torch
from backends.backend_factory import get_backend

class PWLS:

#    def __init__(self, radon: Radon,
#                       sparsifier: TotalVariation):
    def __init__(self, systmatrix: Operator,
                       sparsifier: Sparsifier,
                       backend_name="numpy"):   
       self.systmatrix = systmatrix
       self.sparsifier = sparsifier
       self.backend = get_backend(backend_name)

    def solve(self, x0, b, beta, n_iter, weights):
        
        xk = x0
        ones = self.backend.ones_like(xk)

        D_rec = self.AT(self.backend.mul(weights, self.A(ones)))
        D_reg = self.RabsT(self.Rabs(ones))

        for _ in range(n_iter):
            resid = self.backend.mul(weights, self.backend.add(self.A(xk), -b))
            x_rec = self.backend.add(xk, -self.backend.div(self.AT(resid), D_rec))
            x_reg = self.backend.add(xk, -self.backend.div(self.RT(self.R(xk)), D_reg))

            numerator = self.backend.add(
                self.backend.mul(D_rec, x_rec),
                self.backend.mul(beta * D_reg, x_reg)
            )
            denominator = self.backend.add(D_rec, beta * D_reg)
            xk = self.backend.div(numerator, denominator)
            xk = self.backend.clip_min(xk, 0)

        return xk    
    
    def A(self, x): return self.systmatrix.transform(x)
    def AT(self, x): return self.systmatrix.transposed_transform(x)
    def R(self, x): return self.sparsifier.transform(x)
    def RT(self, x): return self.sparsifier.transposed_transform(x)
    def Rabs(self, x): return self.sparsifier.transform_abs(x)
    def RabsT(self, x): return self.sparsifier.transposed_transform_abs(x)
