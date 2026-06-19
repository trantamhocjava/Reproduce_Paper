model.eval()
with torch.no_grad():
    label_logits, concept_logits = model(img)
