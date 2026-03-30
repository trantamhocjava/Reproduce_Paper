import os
from optparse import OptionParser

import torch
from torch.amp import GradScaler

from . import const, utils


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = const.GPU
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    utils.seed_everything(const.SEEDING)

    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    # Build model
    model = utils.create_model(config)
    model.cuda()

    print("Load img2attr")
    img2attr = utils.load_img2attr(config)

    # Load dataset
    print("Load train, val dataset")
    trainLoader, valLoader, _ = utils.load_train_val_test(
        config, model.preprocess_list, img2attr
    )
    print("Train")
    utils.print_shape_first_batch(trainLoader)
    print("val")
    utils.print_shape_first_batch(valLoader)

    # Initialize covariance with empirical covariance
    if config.cov_type == "empirical":
        model.sigma_concepts = utils.get_empirical_covariance(trainLoader).cuda()
    elif config.cov_type == "global":
        lower_triangle = utils.get_empirical_covariance(trainLoader).cuda()
        rows, cols = torch.tril_indices(
            row=config.num_concepts, col=config.num_concepts, offset=0
        )
        model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        diag_idx = rows == cols
        with torch.no_grad():
            model.sigma_concepts[diag_idx] = (
                lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
            )  # softplus inverse of diag

    loss_fn = const.LOSS[config.model](config=config, reduction="mean")

    optimizer = utils.build_optimizer(model, config)
    scheduler = (
        utils.build_scheduler(optimizer, config) if config.use_scheduler else None
    )
    scaler = GradScaler(const.DEVICE) if config.amp else None

    best_scoring = utils.load_train_state(config, model, optimizer, scaler, scheduler)

    print("Train target predictor")
    train = const.TRAIN[config.model](
        trainLoader,
        valLoader,
        model,
        optimizer,
        scaler,
        "c",
        config,
        loss_fn,
        -1,
    )
    if config.training_mode in ("sequential", "independent"):
        train.mode = "t"
        train.model.freeze_t()
    elif config.training_mode == "joint":
        train.mode = "j"

    utils.train_model(train, config, scheduler, best_scoring)

    print("Evaluate best_model")
    ckpt = torch.load(f"{const.CP_PATH}/best_model.pth", weights_only=False)
    print("train")
    utils.print_dict(ckpt["metric"])
    print("val")
    utils.print_dict(ckpt["val_metric"])

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--last_state",
        type="str",
        dest="last_state",
        default=None,
    )
    parser.add_option(
        "--model",
        type="str",
        dest="model",
    )
    parser.add_option(
        "--best_model_criteria",
        type="str",
        dest="best_model_criteria",
    )
    parser.add_option(
        "--epochs",
        dest="epochs",
        type="int",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option(
        "--optimizer", dest="optimizer", type="str", help="[sgd, adam, adamw]"
    )
    parser.add_option("--lr", dest="lr", type="float")
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
    )
    parser.add_option("--dataset_dir", type="str", dest="dataset_dir")
    parser.add_option("--amp", action="store_true", dest="amp")
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option(
        "--scheduler",
        type="str",
        dest="scheduler",
        default=None,
        help="[LinearLR, ReduceLROnPlateau]",
    )
    parser.add_option(
        "--num_concepts",
        type="int",
        dest="num_concepts",
    )
    parser.add_option(
        "--head_arch", type="str", dest="head_arch", help="[linear, nonlinear]"
    )
    parser.add_option("--alpha", type="float", dest="alpha", default=None)
    parser.add_option(
        "--encoder_arch",
        type="str",
        dest="encoder_arch",
        help="[resnet18, simple_CNN, FCNN]",
    )
    parser.add_option(
        "--training_mode",
        type="str",
        dest="training_mode",
        help="[joint, sequential, independent]",
    )
    parser.add_option(
        "--decrease_every",
        type="int",
        dest="decrease_every",
    )
    parser.add_option(
        "--lr_divisor",
        type="int",
        dest="lr_divisor",
    )
    parser.add_option(
        "--weight_decay",
        type="float",
        dest="weight_decay",
    )
    parser.add_option(
        "--compile",
        action="store_true",
        dest="compile",
    )
    parser.add_option(
        "--num_monte_carlo",
        type="int",
        dest="num_monte_carlo",
    )
    parser.add_option(
        "--straight_through",
        action="store_true",
        dest="straight_through",
    )
    parser.add_option(
        "--concept_learning", type="str", dest="concept_learning", default=None
    )
    parser.add_option("--inter_policy", type="str", dest="inter_policy", default=None)
    parser.add_option(
        "--inter_strategy", type="str", dest="inter_strategy", default=None
    )
    parser.add_option(
        "--pretrain_concepts", action="store_true", dest="pretrain_concepts"
    )
    parser.add_option(
        "--embedding_size", type="int", dest="embedding_size", default=None
    )
    parser.add_option("--cov_type", type="str", dest="cov_type", default=None)
    parser.add_option("--reg_precision", type="str", dest="reg_precision", default=None)
    parser.add_option("--reg_weight", type="float", dest="reg_weight", default=None)
    parser.add_option("--level", type="float", dest="level", default=None)

    (cfg, args) = parser.parse_args()

    main(cfg)
