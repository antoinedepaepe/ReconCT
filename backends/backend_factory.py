def get_backend(name: str = "numpy"):
    if name == "numpy":
        from backends.numpy_backend import get_backend
    elif name == "torch":
        from backends.torch_backend import get_backend
    else:
        raise ValueError(f"Unsupported backend: {name}")
    return get_backend()

