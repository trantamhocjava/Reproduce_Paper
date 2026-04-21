from torch.utils.data import Dataset


class CustomConceptDataset(Dataset):
    def __init__(self, concepts):
        self.concepts = concepts

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        concept = self.concepts[idx]

        return concept
