param_list = []

param_list.append(self.visual_tokens)
for param in self.cross_attn.parameters():
    param_list.append(param)


param_list = []
param_list.append(self.visual_tokens)
param_list.extend(self.cross_attn.parameters())
