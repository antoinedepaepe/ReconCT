def get_backend():
    import numpy as np

    class NumpyBackend:
        def ones_like(self, x): return np.ones_like(x)
        def mul(self, a, b): return np.multiply(a, b)
        def div(self, a, b): return np.divide(a, b)
        def add(self, a, b): return np.add(a, b)
        def clip_min(self, x, min_val): return np.maximum(x, min_val)
        def name(self): return "numpy"

    return NumpyBackend()

def get_backend():
    import numpy as np

    class NumpyBackend:
        def ones_like(self, x): return np.ones_like(x)
        def mul(self, a, b): return np.multiply(a, b)
        def div(self, a, b): return np.divide(a, b)
        def add(self, a, b): return np.add(a, b)
        def clip_min(self, x, min_val): return np.maximum(x, min_val)

        def roll(self, x, shifts, dims):
            for shift, axis in zip(shifts, dims):
                x = np.roll(x, shift=shift, axis=axis)
            return x
        def cat(self, tensors, dim): return np.concatenate(tensors, axis=dim)
        def norm(self, x): return np.linalg.norm(x)
        def rand(self, *shape): return np.random.rand(*shape)
        
        def name(self): return "numpy"

    return NumpyBackend()