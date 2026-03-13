import glob
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from datasets import get_datamodule_fromconfig
from models.conceptBank import get_concept_bank_fromconfig
from utils.utils import config_logging
from models.cbms import LinearCBM  # Đã import sẵn ModelClass ở đây


def get_args():
    parser = argparse.ArgumentParser(description="Huấn luyện HybridCBM bằng PyTorch thuần")

    # 1. THIẾT LẬP CƠ BẢN VÀ ĐƯỜNG DẪN
    parser.add_argument('--dataset', type=str, default='HAM10000', help='Tên bộ dữ liệu')
    parser.add_argument('--data_root', type=str, default='/kaggle/input/ham10000', help='Đường dẫn tới data')
    parser.add_argument('--exp_root', type=str, default='/kaggle/working/experiments/run_1', help='Thư mục lưu weights và log')
    parser.add_argument('--device', type=str, default='cuda:0', help='Thiết bị chạy (cuda:0 hoặc cpu)')
    parser.add_argument('--test', action='store_true', default=False, help='Bật chế độ chỉ test')
    parser.add_argument('--multi', action='store_true', default=False)

    # 2. CẤU HÌNH MÔ HÌNH (CLIP & TRANSLATOR)
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='Tên mô hình CLIP')
    parser.add_argument('--translator_path', type=str, 
                        default='/kaggle/working/weights/translator-AddData/ViT-L_14-AUG_False/translators-best.pt', 
                        help='Đường dẫn tới file weights của Translator')
    parser.add_argument('--weight_init_method', type=str, default='random', help='Cách khởi tạo trọng số')

    # THÊM MỚI: CÁC THAM SỐ BẮT BUỘC MÀ CLASS 'CBM' ĐANG YÊU CẦU
    parser.add_argument('--train_mode', type=str, default='joint', help='Chế độ train: joint, concept_stop_X, ...')
    parser.add_argument('--scale', type=float, default=10.0, help='Hệ số scale cho similarity score')
    parser.add_argument('--use_normalize', action='store_true', default=False, help='Chuẩn hóa vector trước khi phân loại')
    parser.add_argument('--concept_lr', type=float, default=0.001, help='Learning rate riêng cho Dynamic Concepts')
    parser.add_argument('--use_last_ckpt', action='store_true', default=False, help='Resume từ checkpoint gần nhất')

    # 3. THÔNG SỐ KHÁI NIỆM (CONCEPT BOTTLENECK)
    parser.add_argument('--num_class', type=int, default=7, help='Số lượng nhãn/lớp')
    parser.add_argument('--num_static_concept', type=int, default=50, help='Số lượng khái niệm tĩnh')
    parser.add_argument('--num_dynamic_concept', type=int, default=50, help='Số lượng khái niệm động')
    parser.add_argument('--concept_select_fn', type=str, default='submodular', choices=['random', 'submodular'])

    # 4. SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
    parser.add_argument('--max_epochs', type=int, default=50, help='Số vòng huấn luyện tối đa')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate cho Classifier')
    parser.add_argument('--n_shots', type=str, default='all', help='Số lượng mẫu train')

    # 5. TRỌNG SỐ CÁC HÀM LOSS (LAMBDAS)
    parser.add_argument('--lambda_l1', type=float, default=0.0)
    parser.add_argument('--lambda_discri', type=float, default=0.0)
    parser.add_argument('--lambda_discri_alpha', type=float, default=2.0, help='Hệ số alpha cho Discriminability')
    parser.add_argument('--lambda_discri_beta', type=float, default=0.1, help='Hệ số beta cho Discriminability')
    parser.add_argument('--lambda_ort', type=float, default=0.0)
    parser.add_argument('--lambda_align', type=float, default=0.0)

    # BỔ SUNG: THÔNG SỐ DATALOADER VÀ CACHE
    parser.add_argument('--batch_size', type=int, default=128, help='Kích thước batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Số luồng CPU load data')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Ghim bộ nhớ để chuyển GPU nhanh hơn')
    parser.add_argument('--use_img_features', action='store_true', default=True, help='Dùng vector nén thay vì ảnh gốc')
    parser.add_argument('--force_compute', action='store_true', default=False, help='Ép tính toán lại CLIP features từ đầu')
    
    # BỔ SUNG VÀO NHÓM CONCEPT BOTTLENECK:
    parser.add_argument('--submodular_weights', 
                        type=float, 
                        nargs='+', # Bí quyết để nhận mảng nằm ở đây!
                        default=[1.0, 1.0], # Giá trị mặc định là một mảng
                        help='Mảng trọng số cho thuật toán submodular')

    args = parser.parse_args()
    args.exp_root = Path(args.exp_root)
    args.data_root = Path(args.data_root)
    os.makedirs(args.exp_root, exist_ok=True)
    return args


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_data_bank(args, captions=None):
    device = torch.device(args.device)
    concept_bank = get_concept_bank_fromconfig(args)
    
    concept_bank.to(device)
    datamodule = get_datamodule_fromconfig(args, clip_encoder=concept_bank.clip_encoder)
    
    if hasattr(datamodule, 'setup'):
        datamodule.setup(stage='fit')
        
    concept_bank.initialize(img_features=datamodule.img_features['train'],
                            num_images_per_class=datamodule.num_images_per_class,
                            captions=captions)
    concept_bank.to(device)
    return concept_bank, datamodule


def load_checkpoint(args, force_last=False):
    checkpoint_dir = os.path.join(args.exp_root, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None

    if args.use_last_ckpt or force_last:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        return max(checkpoints, key=os.path.getctime) if checkpoints else None
    else:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*val_acc*.pt'))
        if not checkpoints:
            return None
        def get_val_acc(ckpt):
            try:
                return float(os.path.basename(ckpt).split('val_acc=')[-1].split('.pt')[0].split('-')[0])
            except ValueError:
                return 0.0
        return max(checkpoints, key=get_val_acc)


def save_checkpoint(model, args, epoch, step, val_acc, top_k=3):
    checkpoint_dir = os.path.join(args.exp_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*val_acc*.pt'))
    if len(checkpoints) >= top_k:
        checkpoints.sort(key=lambda x: float(x.split('val_acc=')[-1].split('.pt')[0].split('-')[0]))
        os.remove(checkpoints[0]) 
        
    save_path = os.path.join(checkpoint_dir, f'epoch={epoch}-step={step}-val_acc={val_acc:.4f}.pt')
    torch.save(model.state_dict(), save_path)


def train(args, ModelClass, captions=None):
    seed_everything(42)
    device = torch.device(args.device)
    
    concept_bank, datamodule = init_data_bank(args, captions)

    logger = config_logging(log_file=f'{args.exp_root}/train.log')
    tb_logger = SummaryWriter(log_dir=os.path.join(args.exp_root, f'{args.dataset}_{args.n_shots}shot_train'))
    
    model = ModelClass(args, conceptbank=concept_bank)

    ckpt_path = load_checkpoint(args, force_last=True)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        logger.info(f"Đã khôi phục trọng số từ: {ckpt_path}")

    model.to(device)

    # ĐÃ SỬA: KHỞI TẠO 2 OPTIMIZER CHO CONCEPT VÀ CLASSIFIER RIÊNG BIỆT
    opt_dynamic = torch.optim.Adam([
        {'params': model.conceptbank.dynamic_bank.parameters(), 'lr': args.concept_lr},
    ])
    opt_classifier = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': [model.scale], 'lr': args.lr},
    ])

    if hasattr(datamodule, 'setup'): datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    check_interval = 5 if (args.dataset == "ImageNet" and args.n_shots == "all") else 10
    global_step = 0
    best_val_acc = 0.0

    logger.info("BẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(args.max_epochs):
        model.train()
        model.current_epoch = epoch  # Cập nhật epoch cho CBM class
        
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}")

        for batch in progress_bar:
            images, labels = batch[0].to(device), batch[1].to(device)
            
            opt_dynamic.zero_grad()
            opt_classifier.zero_grad()
            
            # ĐÃ SỬA: GỌI ĐÚNG HÀM TRAIN_CONCEPT VÀ TRAIN_CLASSIFIER
            final_loss = 0
            if model.is_train_concept:
                final_loss += model.train_concept(images, labels)
                
            if model.is_train_cls:
                final_loss += model.train_classifier(images, labels)
            
            # Lan truyền ngược
            if torch.is_tensor(final_loss):
                final_loss.backward()
            
            # Cập nhật trọng số
            if model.is_train_concept:
                opt_dynamic.step()
            opt_classifier.step()
            
            running_loss += final_loss.item() if torch.is_tensor(final_loss) else final_loss
            global_step += 1
            progress_bar.set_postfix({'loss': f'{running_loss/global_step:.4f}'})
            
        tb_logger.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)

        # VALIDATION PHASE
        if epoch % check_interval == 0 or epoch == args.max_epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            val_acc = correct / total
            tb_logger.add_scalar('Accuracy/Validation', val_acc, epoch)
            logger.info(f"Epoch {epoch} | Val Acc: {val_acc:.4f}")
            
            save_checkpoint(model, args, epoch, global_step, val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    tb_logger.close()


def test(args, ModelClass, captions=None): # ĐÃ SỬA: Thêm tham số truyền vào
    seed_everything(42)
    device = torch.device(args.device)
    
    concept_bank, datamodule = init_data_bank(args, captions)
    logger = config_logging(log_file=f'{args.exp_root}/test.log')
    
    ckpt_path = load_checkpoint(args) 
    logger.info(f"Đang tải checkpoint tốt nhất từ: {ckpt_path}")
    
    model = ModelClass(args, conceptbank=concept_bank)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        
    model.to(device)
    model.eval()
    
    if hasattr(datamodule, 'setup'): datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = 100 * correct / total
    logger.info(f'Kết quả Test Accuracy cuối cùng: {test_acc:.2f}%')
    
    # ĐÃ SỬA: GỌI HÀM EXPLAINABLE AI SAU KHI TEST XONG
    if hasattr(model, 'save_topk_concepts_for_class'):
        logger.info("Đang trích xuất và lưu file CSV các khái niệm đặc trưng cho Y Khoa (Explainability)...")
        model.save_topk_concepts_for_class()


def main():
    args = get_args()
    ModelClass = LinearCBM

    if not args.test:
        train(args, ModelClass)
        test(args, ModelClass)
    else:
        test(args, ModelClass)


if __name__ == "__main__":
    main()