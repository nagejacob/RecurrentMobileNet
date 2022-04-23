def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    print(params)
    return params