class HybridCBMTrain(pl.LightningModule):
    def __init__(self, config, select_concept_data):
        super().__init__()

        self.config = config

        # Model
        self.hybrid_bank = HybridConceptBank(config, select_concept_data)
        self.hybridcbm = HybridCBM(
            config=config, concept_feat=self.hybrid_bank.concept_feat
        )

    # define optimizers and schedulers
    def configure_optimizers(self):
        optimizer_dynamic_concept = kltn_utils.build_optimizer(
            self.hybrid_bank.dynamic_bank,
            self.config.optimizer_dynamic_concept,
        )
        optimizer_hybridcbm = kltn_utils.build_optimizer(
            self.hybridcbm,
            self.config.optimizer_hybridcbm,
        )

        return [optimizer_dynamic_concept, optimizer_hybridcbm]

    def training_step(self, batch, batch_idx):
        result = self.get_loss(batch)

        # Update optimizer
        self.manual_backward(result["loss"])

        opt_dynamic_concept, opt_classifier = self.optimizers()

        kltn_utils.update_optimizer(opt_dynamic_concept)
        kltn_utils.update_optimizer(opt_classifier)

        # Update loss and metric
        self.train_metric.update(result)
