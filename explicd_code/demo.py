concept_list = explicid_isic_dict
net = ExpLICD(concept_list=concept_list, model_name="biomedclip", config=config)

# We find using orig_in21k vit weights works better than biomedclip vit weights
# Delete the following if want to use biomedclip weights
vit = timm.create_model(
    "vit_base_patch16_224.orig_in21k", pretrained=True, num_classes=config.num_class
)
vit.head = nn.Identity()
net.model.visual.trunk.load_state_dict(vit.state_dict())

net.load_state_dict(torch.load(config.load))
print("Model loaded from {}".format(config.load))

net.cuda()
