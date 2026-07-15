import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B,L,D), got {x.shape}")
        B, L, D = x.shape
        positions = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, L)
        pos = self.pos_embedding(positions)
        return x + pos

class TransformerSleepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_heads=8, num_layers=3, num_classes=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, L, w, F) -> (B, L, w*F)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)        # (B, L, H)
        x = self.pos_encoder(x)       # (B, L, H)
        x = self.transformer_encoder(x)
        return x                      # (B, L, H)

# class TwoHeadSleepModel(nn.Module):
#     def __init__(self, feature_extractor,classes=2, depth=1, hidden_dim=512):
#         super().__init__()
#         self.feature_extractor = feature_extractor  # 冻结 backbone

#         self.nerm_ff = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, 2048),
#             nn.GELU(),
#             nn.Linear(2048, hidden_dim),
#         )
#         self.nerm_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim,classes)
#         )

#         self.wake_ff = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, 2048),
#             nn.GELU(),
#             nn.Linear(2048, hidden_dim),
#         )
#         self.wake_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim,classes),
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)             # (B, L, H)
#         nerm_feat = self.nerm_ff(features) + features    # 残差
#         wake_feat = self.wake_ff(features) + features  # 残差
#         return self.nerm_head(nerm_feat), self.wake_head(wake_feat)  # (B,L,2), (B,L,2)
class ThreeHeadSleepModel(nn.Module):
    def __init__(self, feature_extractor, classes=2, hidden_dim=512):
        super().__init__()
        self.feature_extractor = feature_extractor  # 冻结 backbone
        
        # --- 原始 NREM/REM 分类头 ---
        self.nerm_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, hidden_dim),
        )
        self.nerm_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classes) # classes=2, 假设用于 NREM/REM 分类 Logits
        )

        # --- 原始 Wake 分类头 ---
        self.wake_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, hidden_dim),
        )
        self.wake_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, classes), # classes=2, 假设用于 Wake/Non-Wake 分类 Logits
        )

        # --- 新增的 Wake/Non-Wake 深度回归头 ---
        self.wake_depth_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, hidden_dim),
        )
        # 输出维度为 1，用于连续的深度得分 D_pred (Logits 式尺度)
        self.wake_depth_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1), 
        )
        #REM深度
        # self.REM_depth_ff=nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, 2048),
        #     nn.GELU(),
        #     nn.Linear(2048, hidden_dim),
        # )
        # self.REM_depth_head=nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, 1), 
        # )

    def forward(self, x):
        features = self.feature_extractor(x)           # (B, L, H)
        
        # --- NREM/REM 分类分支 ---
        nerm_feat = self.nerm_ff(features) + features  # 残差连接
        nerm_logits = self.nerm_head(nerm_feat)        # (B, L, classes)

        # --- Wake 分类分支 ---
        wake_feat = self.wake_ff(features) + features  # 残差连接
        wake_logits = self.wake_head(wake_feat)        # (B, L, classes)
        
        # --- Wake/Non-Wake 深度回归分支 (新增) ---
        wake_depth_feat = self.wake_depth_ff(features) + features # 残差连接
        # wake_depth_score 是连续的深度值 D_pred (B, L, 1)
        wake_depth_score = self.wake_depth_head(wake_depth_feat) 
        #  REM深度回归分支
        # rem_depth_feat=self.REM_depth_ff(features) + features
        # rem_depth_score=self.REM_depth_head(rem_depth_feat)
        
        # 返回三个头部的输出
        # nerm_logits: NREM/REM Logits
        # wake_logits: Wake/Non-Wake Logits
        # wake_depth_score: 连续的 Wake/Non-Wake 深度得分 D_pred
        return nerm_logits, wake_logits, wake_depth_score#,rem_depth_score
