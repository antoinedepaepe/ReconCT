from math import sqrt

class Gradient:
    def __init__(self, weight: str = 'standard', backend=None):
        if backend is None:
            raise ValueError("Un backend est requis (torch ou numpy)")
        self.backend = backend

        if weight == 'standard':
            self.diag_weights = sqrt(1/2)
        elif weight == 'sqrt':
            self.diag_weights = sqrt(sqrt(1/2))
        else:
            raise ValueError("Poids inconnu : utilisez 'standard' ou 'sqrt'")

    def transform(self, x):
        b = self.backend
        f = -1.0

        x1 = x + f * b.roll(x, shifts=(0, 1), dims=(2, 3))
        x2 = x + f * b.roll(x, shifts=(0, -1), dims=(2, 3))
        x3 = x + f * b.roll(x, shifts=(1, 0), dims=(2, 3))
        x4 = x + f * b.roll(x, shifts=(-1, 0), dims=(2, 3))

        x5 = self.diag_weights * (x + f * b.roll(x, shifts=(1, -1), dims=(2, 3)))
        x6 = self.diag_weights * (x + f * b.roll(x, shifts=(-1, 1), dims=(2, 3)))
        x7 = self.diag_weights * (x + f * b.roll(x, shifts=(1, 1), dims=(2, 3)))
        x8 = self.diag_weights * (x + f * b.roll(x, shifts=(-1, -1), dims=(2, 3)))

        return b.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

    def transposed_transform(self, x, factor: float = -1.0):
        b = self.backend

        x1 = x[:, 0:1, :, :] + factor * b.roll(x[:, 0:1, :, :], shifts=(0, -1), dims=(2, 3))
        x2 = x[:, 1:2, :, :] + factor * b.roll(x[:, 1:2, :, :], shifts=(0, 1), dims=(2, 3))
        x3 = x[:, 2:3, :, :] + factor * b.roll(x[:, 2:3, :, :], shifts=(-1, 0), dims=(2, 3))
        x4 = x[:, 3:4, :, :] + factor * b.roll(x[:, 3:4, :, :], shifts=(1, 0), dims=(2, 3))

        x5 = self.diag_weights * (x[:, 4:5, :, :] + factor * b.roll(x[:, 4:5, :, :], shifts=(-1, 1), dims=(2, 3)))
        x6 = self.diag_weights * (x[:, 5:6, :, :] + factor * b.roll(x[:, 5:6, :, :], shifts=(1, -1), dims=(2, 3)))
        x7 = self.diag_weights * (x[:, 6:7, :, :] + factor * b.roll(x[:, 6:7, :, :], shifts=(-1, -1), dims=(2, 3)))
        x8 = self.diag_weights * (x[:, 7:8, :, :] + factor * b.roll(x[:, 7:8, :, :], shifts=(1, 1), dims=(2, 3)))

        return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

    def transform_abs(self, x, factor: float = 1.0):
        b = self.backend

        x1 = x + factor * b.roll(x, shifts=(0, 1), dims=(2, 3))
        x2 = x + factor * b.roll(x, shifts=(0, -1), dims=(2, 3))
        x3 = x + factor * b.roll(x, shifts=(1, 0), dims=(2, 3))
        x4 = x + factor * b.roll(x, shifts=(-1, 0), dims=(2, 3))

        x5 = self.diag_weights * (x + factor * b.roll(x, shifts=(1, -1), dims=(2, 3)))
        x6 = self.diag_weights * (x + factor * b.roll(x, shifts=(-1, 1), dims=(2, 3)))
        x7 = self.diag_weights * (x + factor * b.roll(x, shifts=(1, 1), dims=(2, 3)))
        x8 = self.diag_weights * (x + factor * b.roll(x, shifts=(-1, -1), dims=(2, 3)))

        return b.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)

    def transposed_transform_abs(self, x, factor: float = 1.0):
        return self.transposed_transform(x, factor=factor)

    def norm(self, M: int, N: int, max_iter: int = 50):
        b = self.backend
        x = b.rand(1, 1, M, N)
        s = 0
        for _ in range(max_iter):
            x, s = self.norm_one_step(x, s)
        return s

    def norm_one_step(self, x, s):
        b = self.backend
        x = self.transposed_transform(self.transform(x))
        x = b.div(x, b.norm(x))
        s = b.norm(self.transform(x))
        return x, s