from optparse import OptionParser

from kltn_utils import kltn_const, kltn_utils

from ...models.explicd_hybridcbm.train import ExplicdHybridCBMTrain
from ...models.explicd_hybridcbm.train_v1 import ExplicdHybridCBMTrain_v1
from ...models.hybridcbm.train import MetricCalculator
from ..train_hybridcbm import train_hybridcbm


class ExplicdHybridCBMTrainer(train_hybridcbm.HybridCBMTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        img_embed_dim = kltn_const.CLIP_MODELS[config.model.clip_model]["embedding_dim"]
        concept_feat_dim = self.select_concept_data["concept_feat"].shape[1]

        if img_embed_dim == concept_feat_dim:
            self.model = ExplicdHybridCBMTrain(
                CustomMetric=MetricCalculator,
                cp_path=config.cp_path,
                config=config,
                select_concept_data=self.select_concept_data,
            )
        else:
            self.model = ExplicdHybridCBMTrain_v1(
                CustomMetric=MetricCalculator,
                cp_path=config.cp_path,
                config=config,
                select_concept_data=self.select_concept_data,
                img_embed_dim=img_embed_dim,
                concept_feat_dim=concept_feat_dim,
            )


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    ExplicdHybridCBMTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
