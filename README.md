# Purpose of the project

- Developed Concept-based Models for medical image disease classification tasks with interpretable decision explanations.
- Researched and implemented state-of-the-art papers related to Explainable AI (XAI).
- Experimented with existing approaches and proposed novel model ideas to improve overall model performance.

# State-of-the-art papers

- Concept Bottleneck Models ([Paper link](https://arxiv.org/pdf/2007.04612))
- A Comprehensive Survey on the Risks and Limitations of Concept-based Models ([Paper link](https://arxiv.org/pdf/2506.04237v1))
- Aligning Human Knowledge with Visual Concepts Towards Explainable Medical Image Classification ([Paper link](https://arxiv.org/pdf/2406.05596))
- AdaCBM: An Adaptive Concept Bottleneck Model for Explainable and Accurate Diagnosis ([Paper link](https://arxiv.org/pdf/2408.02001))
- Hybrid Concept Bottleneck Models ([Paper link](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hybrid_Concept_Bottleneck_Models_CVPR_2025_paper.pdf?utm_source=chatgpt.com))
- Energy-based Concept Bottleneck Models (ECBM) ([Paper link](https://arxiv.org/pdf/2401.14142v4))
- Stochastic Concept Bottleneck Models ([Paper link](https://arxiv.org/pdf/2406.19272))

# Datasets

- For training, we split the original dataset into train, validation, test with the ratio below, and push to Kaggle for experiments:
  - Train: 60 %
  - Validation: 20 %
  - Test: 20 %
- Datasets:
  - ISIC2018 ([Dataset link](https://www.kaggle.com/datasets/tmtrnhelloworld/isic2018splittedv1))
  - BUSI ([Dataset link](https://www.kaggle.com/datasets/tmtrnhelloworld/busikeepimage))
  - IDRID ([Dataset link](https://www.kaggle.com/datasets/tmtrnhelloworld/idridsplittedkeepimage))
  - Lung Colon Cancer Dataset ([Dataset link](https://www.kaggle.com/datasets/tmtrnhelloworld/lungcoloncancersplitted))
  - NCT-CRC-HE10K ([Dataset link](https://www.kaggle.com/datasets/tmtrnhelloworld/nct-crc-he10ksplitted))

# Our ideas

- Used Hybrid Concept Bottleneck Models as the foundation for development, incorporating the loss functions proposed in the original paper.

- Experimented with integrating the adaptive module from AdaCBM and the Cross-Attention mechanism from ExpLICD into the base model architecture.

- Evaluated the proposed model against established baselines, including AdaCBM, ExpLICD, SCBM, and ECBM.
