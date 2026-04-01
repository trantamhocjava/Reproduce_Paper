def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    for param in m.parameters():
        param.requires_grad = True
