tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
self.model.eval()
with torch.no_grad():
    _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
self.concept_token_dict[key] = tmp_concept_feats
