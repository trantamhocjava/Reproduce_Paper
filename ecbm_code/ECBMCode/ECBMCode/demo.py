for param in self.backbone.parameters():
    param.requires_grad = False
