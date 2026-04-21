def configure_optimizers(self):
    optimizer = kltn_utils.build_optimizer(self.translator, self.config)
    lr_scheduler, monitor = get_linear_schedule_with_warmup(
        optimizer,  # Optimizer đang sử dụng (ví dụ: AdamW)
        num_warmup_steps=0,  # Số bước warmup (thường là khoảng 0-10% tổng số bước huấn luyện)
        num_training_steps=1000,  # Tổng số bước huấn luyện (epoch * bước / batch)
    )
    res = {
        "optimizer": optimizer,
    }

    if lr_scheduler is not None:
        res["lr_scheduler"] = lr_scheduler

    if monitor is not None:
        res["monitor"] = monitor

    return res


class MetricCalculator:
    def reset(self):
        self.loss = 0
        self.loss_token = 0
        self.epoch_acc = 0

        self.n_batchs = 0
