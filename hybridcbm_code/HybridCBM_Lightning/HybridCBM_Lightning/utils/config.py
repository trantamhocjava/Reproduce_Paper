import argparse
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="HybridCBM Training Options")

    # 1. THIẾT LẬP CƠ BẢN VÀ ĐƯỜNG DẪN
    parser.add_argument(
        "--dataset", type=str, default="HAM10000", help="Tên bộ dữ liệu"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/ham10000",
        help="Đường dẫn tới data",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="/kaggle/working/experiments/run_1",
        help="Thư mục lưu weights và log",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Thiết bị chạy (cuda:0 hoặc cpu)"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Bật chế độ chỉ test"
    )
    parser.add_argument("--multi", action="store_true", default=False)

    # 2. CẤU HÌNH MÔ HÌNH (CLIP & TRANSLATOR)
    parser.add_argument(
        "--clip_model", type=str, default="ViT-L/14", help="Tên mô hình CLIP"
    )
    parser.add_argument(
        "--translator_path",
        type=str,
        default="/kaggle/working/weights/translator-AddData/ViT-L_14-AUG_False/translators-best.pt",
        help="Đường dẫn tới file weights của Translator",
    )
    parser.add_argument(
        "--weight_init_method",
        type=str,
        default="random",
        help="Cách khởi tạo trọng số",
    )

    # THÊM MỚI: CÁC THAM SỐ BẮT BUỘC MÀ CLASS 'CBM' ĐANG YÊU CẦU
    parser.add_argument(
        "--train_mode",
        type=str,
        default="joint",
        help="Chế độ train: joint, concept_stop_X, ...",
    )
    parser.add_argument(
        "--scale", type=float, default=10.0, help="Hệ số scale cho similarity score"
    )
    parser.add_argument(
        "--use_normalize",
        action="store_true",
        default=False,
        help="Chuẩn hóa vector trước khi phân loại",
    )
    parser.add_argument(
        "--concept_lr",
        type=float,
        default=0.001,
        help="Learning rate riêng cho Dynamic Concepts",
    )
    parser.add_argument(
        "--use_last_ckpt",
        action="store_true",
        default=False,
        help="Resume từ checkpoint gần nhất",
    )

    # 3. THÔNG SỐ KHÁI NIỆM (CONCEPT BOTTLENECK)
    parser.add_argument("--num_class", type=int, default=7, help="Số lượng nhãn/lớp")
    parser.add_argument(
        "--num_static_concept", type=int, default=50, help="Số lượng khái niệm tĩnh"
    )
    parser.add_argument(
        "--num_dynamic_concept", type=int, default=50, help="Số lượng khái niệm động"
    )
    parser.add_argument(
        "--concept_select_fn",
        type=str,
        default="submodular",
        choices=["random", "submodular"],
    )

    # 4. SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Số vòng huấn luyện tối đa"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate cho Classifier"
    )
    parser.add_argument("--n_shots", type=str, default="all", help="Số lượng mẫu train")

    # 5. TRỌNG SỐ CÁC HÀM LOSS (LAMBDAS)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_discri", type=float, default=0.0)
    parser.add_argument(
        "--lambda_discri_alpha",
        type=float,
        default=2.0,
        help="Hệ số alpha cho Discriminability",
    )
    parser.add_argument(
        "--lambda_discri_beta",
        type=float,
        default=0.1,
        help="Hệ số beta cho Discriminability",
    )
    parser.add_argument("--lambda_ort", type=float, default=0.0)
    parser.add_argument("--lambda_align", type=float, default=0.0)

    # BỔ SUNG: THÔNG SỐ DATALOADER VÀ CACHE
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Kích thước batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Số luồng CPU load data"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Ghim bộ nhớ để chuyển GPU nhanh hơn",
    )
    parser.add_argument(
        "--use_img_features",
        action="store_true",
        default=True,
        help="Dùng vector nén thay vì ảnh gốc",
    )
    parser.add_argument(
        "--force_compute",
        action="store_true",
        default=False,
        help="Ép tính toán lại CLIP features từ đầu",
    )

    # BỔ SUNG VÀO NHÓM CONCEPT BOTTLENECK:
    parser.add_argument(
        "--submodular_weights",
        type=float,
        nargs="+",  # Bí quyết để nhận mảng nằm ở đây!
        default=[1.0, 1.0],  # Giá trị mặc định là một mảng
        help="Mảng trọng số cho thuật toán submodular",
    )

    # BỔ SUNG CHO ADAPTIVE MODULE (nếu cần)
    parser.add_argument(
        "--adaptive_num_layers",
        type=int,
        default=2,
        help="Số lớp trong Adaptive Module",
    )

    parser.add_argument(
        "--adaptive_residual",
        action="store_true",
        default=False,
        help="Có sử dụng kết nối residual trong Adaptive Module hay không",
    )

    parser.add_argument(
        "--adaptive_use_img_norm",
        action="store_true",
        default=False,
        help="Có chuẩn hóa đặc trưng ảnh trong Adaptive Module hay không",
    )

    args = parser.parse_args()
    args.exp_root = Path(args.exp_root)
    args.data_root = Path(args.data_root)
    os.makedirs(args.exp_root, exist_ok=True)
    return args
