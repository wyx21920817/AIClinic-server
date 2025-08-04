
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))




# import os
# import sys
# import torch
# import random
# import numpy as np
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
# import wfdb
# from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from code.models.xresnet1d import xresnet1d101
# from sklearn.model_selection import train_test_split

# # ------------------- 配置 -------------------
# DATA_CSV = 'output/exp0/data/ptbxl_database.csv'  # ECG record info
# RECORD_ROOT = 'data/ptbxl/records100'            # WFDB records base path
# SCALER_PATH = 'output/exp0/data/standard_scaler.pkl'
# MLB_PATH    = 'output/exp0/data/mlb.pkl'
# WEIGHTS_PTH = 'output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth'
# BATCH_SIZE  = 16
# LR_HEAD     = 1e-3
# LR_BACKBONE= 1e-4
# EPOCHS      = 10
# SEED        = 42
# DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# # ------------------- 工具函数 -------------------
# def set_seed(seed=SEED):
#     random.seed(seed);
#     np.random.seed(seed);
#     torch.manual_seed(seed);
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# def load_scaler_and_mlb(scaler_path=SCALER_PATH, mlb_path=MLB_PATH):
#     import pickle
#     with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
#     with open(mlb_path, 'rb') as f: mlb = pickle.load(f)
#     return scaler, mlb


# def load_pretrained_model(weights_path, n_in, n_out, hidden_size=None):
#     # 若已知 hidden_size，请传入；否则自动推断。
#     ckpt = torch.load(weights_path, map_location=DEVICE)
#     state = ckpt.get('model', ckpt)
#     if hidden_size is None:
#         for k,v in state.items():
#             if v.ndim==2 and v.shape[0]==n_out:
#                 hidden_size = v.shape[1]; break
#         assert hidden_size, '无法推断 hidden_size'
#     model = xresnet1d101(
#         input_channels=n_in,
#         num_classes=n_out,
#         lin_ftrs_head=[hidden_size]
#     ).to(DEVICE)
#     model.load_state_dict(state, strict=True)
#     return model

# # ------------------- Dataset -------------------
# class PTBXL_Dataset(Dataset):
#     def __init__(self, df, record_root, scaler, mlb):
#         self.df = df.reset_index(drop=True)
#         self.root = record_root
#         self.scaler = scaler
#         self.mlb = mlb

#     def __len__(self): return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         rec = wfdb.rdrecord(os.path.join(self.root, row['record_name']))
#         sig = rec.p_signal.astype(np.float32).T
#         # 标准化
#         flat = sig.flatten()[:,None]
#         sig = self.scaler.transform(flat).reshape(sig.shape)
#         # 标签 one-hot
#         labels = self.mlb.transform([row['scp_codes_list']])[0]
#         return torch.from_numpy(sig), torch.from_numpy(labels.astype(np.float32))

# # ------------------- 主函数 -------------------
# def main():
#     set_seed()
#     import pandas as pd
#     # 读取元数据
#     df = pd.read_csv(DATA_CSV)
#     # 假设 df 包含 'record_name' 和 'scp_codes_list' 字段，scp_codes_list 是 list 类型
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

#     scaler, mlb = load_scaler_and_mlb()
#     NUM_CLASSES = len(mlb.classes_)
#     model = load_pretrained_model(WEIGHTS_PTH, n_in=12, n_out=NUM_CLASSES)

#     # 冻结主干，仅训练 head
#     for name, p in model.named_parameters():
#         if 'head' not in name:
#             p.requires_grad = False

#     # 优化器：head lr 大，backbone lr 小
#     optimizer = optim.Adam([
#         {'params': model.head.parameters(), 'lr': LR_HEAD},
#         {'params': [p for n,p in model.named_parameters() if 'head' not in n], 'lr': LR_BACKBONE}
#     ])
#     criterion = nn.BCEWithLogitsLoss()

#     train_ds = PTBXL_Dataset(train_df, RECORD_ROOT, scaler, mlb)
#     val_ds   = PTBXL_Dataset(val_df, RECORD_ROOT, scaler, mlb)
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

#     # 训练循环
#     for epoch in range(1, EPOCHS+1):
#         model.train()
#         total_loss=0
#         for x,y in train_loader:
#             x,y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             logits = model(x)
#             loss = criterion(logits, y)
#             loss.backward(); optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {total_loss/len(train_loader):.4f}")

#         # 验证
#         model.eval()
#         val_loss=0
#         with torch.no_grad():
#             for x,y in val_loader:
#                 x,y = x.to(DEVICE), y.to(DEVICE)
#                 logits=model(x)
#                 val_loss += criterion(logits,y).item()
#         print(f"           Val   Loss: {val_loss/len(val_loader):.4f}\n")

#     # 保存微调后的权重
#     torch.save({'model': model.state_dict()}, 'output/exp0/models/finetuned_xresnet1d101.pth')

# if __name__=='__main__':
#     main()
