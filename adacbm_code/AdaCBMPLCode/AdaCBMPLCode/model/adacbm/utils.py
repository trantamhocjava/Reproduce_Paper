import torch


def get_class2concept(num_class, concept2cls, init_val=1):
    num_concept = len(concept2cls)
    concept2cls = torch.from_numpy(concept2cls).long().view(1, -1)
    class2concept = torch.zeros((num_class, num_concept))
    class2concept.scatter_(0, concept2cls, init_val)

    return class2concept
