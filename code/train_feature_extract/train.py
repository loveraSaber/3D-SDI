'''
donglin xie 2025-11-07
train.py
train the model
'''
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from Wake_rankloss import PairMarginRankLoss
from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer import TransformerSleepModel, ThreeHeadSleepModel

# 屏蔽无关紧要的第三方警告
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")

# 参数配置
CONFIG = {
    "pretrained_model_path": "/home/zhaoqingshuo/SDI/ECG_sleepstage/step2_model/all_42_final_model_best.pth",
    "train_list": "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/inner/train_paths.txt",
    "val_list": "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/data_split/inner/val_paths.txt",
    "save_path": "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/",  # 结果保存目录
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-3,
    "window_size": 15,
    "stride": 1,
    "seed": 42,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
    "early_stop_patience": 5,
    "num_classes": 2
}

def load_feature_extractor(pretrained_path, device):
    input_dim = CONFIG["window_size"] * 1152
    feature_extractor = TransformerSleepModel(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"]
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    missing, unexpected = feature_extractor.load_state_dict(new_state_dict, strict=False)
    
    # 打印加载信息
    print(f"Loaded Feature Extractor. missing: {missing} unexpected: {unexpected}")

    feature_extractor.to(device)
    # 冻结特征提取器
    for p in feature_extractor.parameters():
        p.requires_grad = False
    return feature_extractor

def build_four_head_model(pretrained_path, device, local_rank, use_ddp):
    feature_extractor = load_feature_extractor(pretrained_path, device)
    
    model = ThreeHeadSleepModel(
        feature_extractor=feature_extractor,
        classes=2,
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        trainable_params = [p for p in model.module.parameters() if p.requires_grad]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    return model, trainable_params

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_paths_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return f.read().splitlines()

def train_one_epoch(model, loader, optimizer, wake_loss_fn, rem_loss_fn, rank_loss_fn, device, alpha=0.5, beta=0.1, gamma=1.0, is_main=True):
    model.train()
    total_loss = total_rank = total_const = 0.0

    pbar = tqdm(loader, desc="Training", leave=False, disable=not is_main)
    for x, wake_labels, rem_labels, wake_depth in pbar:
        x, wake_labels, rem_labels, wake_depth = [t.to(device, non_blocking=True) for t in [x, wake_labels, rem_labels, wake_depth]]
        
        optimizer.zero_grad()
        
        # 接收模型返回的 4 个输出
        rem_logits, wake_logits, d_wake= model(x) 

        # 展平处理
        wake_logits_flat = wake_logits.view(-1, wake_logits.size(-1))
        rem_logits_flat = rem_logits.view(-1, rem_logits.size(-1))
        wake_labels_flat = wake_labels.view(-1)
        rem_labels_flat = rem_labels.view(-1)
        wake_depth_flat = wake_depth.view(-1) 
        
        # 损失计算
        loss_wake = wake_loss_fn(wake_logits_flat, wake_labels_flat)
        loss_rem = rem_loss_fn(rem_logits_flat, rem_labels_flat)
        loss_rank = rank_loss_fn(d_wake.view(-1), wake_depth_flat) 
        
        # 物理约束损失
        valid_mask = (wake_depth_flat != -100).float()
        margin = 0.5 
        loss_const = (torch.relu(d_wake.view(-1) + margin - d_sleep.view(-1)) * valid_mask).mean()
        
        total_loss_batch = loss_wake + alpha * loss_rem + beta * loss_rank + gamma * loss_const
        
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        total_rank += loss_rank.item()
        total_const += loss_const.item()

        if is_main:
            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1e-9):.4f}", const=f"{total_const/(pbar.n+1e-9):.4f}")

    return total_loss / len(loader)

def validate(model, loader, wake_loss_fn, rem_loss_fn, rank_loss_fn, device, alpha=0.5, beta=0.1, gamma=1.0, is_main=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, disable=not is_main)
        for x, wake_labels, rem_labels, wake_depth in pbar:
            x, wake_labels, rem_labels, wake_depth = [t.to(device) for t in [x, wake_labels, rem_labels, wake_depth]]
            
            rem_logits, wake_logits, d_wake= model(x)
            
            loss_wake = wake_loss_fn(wake_logits.view(-1, 2), wake_labels.view(-1))
            loss_rem = rem_loss_fn(rem_logits.view(-1, 2), rem_labels.view(-1))
            loss_rank = rank_loss_fn(d_wake.view(-1), wake_depth.view(-1))
            valid_mask = (wake_depth.view(-1) != -100).float()
            loss_const = (torch.relu(d_wake.view(-1) + 0.5 - d_sleep.view(-1)) * valid_mask).mean()

            total_loss += (loss_wake + alpha * loss_rem + beta * loss_rank + gamma * loss_const).item()
            
    avg_loss = total_loss / len(loader)
    if is_main: print(f"Val Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    set_seed(CONFIG["seed"])

    # 1. 分布式环境初始化
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = (dist.get_rank() == 0)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    # 2. 确保保存目录存在
    if is_main:
        if not os.path.exists(CONFIG["save_path"]):
            os.makedirs(CONFIG["save_path"])

    # 3. 数据准备
    train_files = load_paths_from_txt(CONFIG["train_list"])
    val_files = load_paths_from_txt(CONFIG["val_list"])

    train_set = SleepWindowDataset(train_files, CONFIG["window_size"], CONFIG["stride"])
    val_set = SleepWindowDataset(val_files, CONFIG["window_size"], CONFIG["stride"])

    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, drop_last=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn_window,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn_window,
        num_workers=4,
        pin_memory=True
    )

    # 4. 初始化模型、优化器与损失函数
    model, trainable_params = build_four_head_model(CONFIG["pretrained_model_path"], device, local_rank, use_ddp)
    optimizer = optim.Adam(trainable_params, lr=CONFIG["lr"])

    wake_loss = nn.CrossEntropyLoss(ignore_index=-100)
    rem_loss = nn.CrossEntropyLoss(ignore_index=-100)
    rank_loss = PairMarginRankLoss() 

    ALPHA = 0.5
    BETA = 0.1
    GAMMA = 1.0 

    # 5. ✨ 关键变量初始化 ✨
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    # 修正 KeyError: 使用 CONFIG["save_path"] 而非 "output_dir"
    best_ckpt_path = os.path.join(CONFIG["save_path"], "best_model.pth")

    # 6. 训练循环
    for epoch in range(CONFIG["epochs"]):
        if use_ddp: 
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, 
            wake_loss, rem_loss, rank_loss, 
            device, ALPHA, BETA, GAMMA, is_main
        )
        
        val_loss = validate(
            model, val_loader, 
            wake_loss, rem_loss, rank_loss, 
            device, ALPHA, BETA, GAMMA, is_main
        )

        # 仅主进程处理保存与早停逻辑
        if is_main:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                state_dict = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save(state_dict, best_ckpt_path)
                print(f"✅ Best model saved at epoch {epoch+1} -> {best_ckpt_path} (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience {patience_counter}/{CONFIG['early_stop_patience']}")

        # 同步各进程的早停状态
        stop_tensor = torch.tensor(
            [1 if patience_counter >= CONFIG["early_stop_patience"] else 0],
            device=device
        )
        if use_ddp:
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)

        if stop_tensor.item() == 1:
            if is_main:
                print(f"🛑 Early stopping at epoch {epoch+1}. Best epoch {best_epoch}, best val {best_loss:.4f}")
            break

    if is_main:
        print(f"🏁 Training finished. Best epoch={best_epoch}, best val_loss={best_loss:.4f}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()