def get_backend():
    import torch

    class TorchBackend:
        def ones_like(self, x): return torch.ones_like(x, device=x.device)
        def mul(self, a, b): return a * b
        def div(self, a, b): return a / b
        def add(self, a, b): return a + b
        def clip_min(self, x, min_val): 
            x = x.clone()
            x[x < min_val] = min_val
            return x

        def roll(self, x, shifts, dims): return torch.roll(x, shifts=shifts, dims=dims)
        def cat(self, tensors, dim): return torch.cat(tensors, dim=dim)
        def norm(self, x): return torch.norm(x)
        def rand(self, *shape): return torch.rand(*shape)
        
        def name(self): return "torch"

    return TorchBackend()

