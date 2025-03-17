from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer


# DecoderLayer（解码器层）：基于多头自注意力（Multi-head Attention）计算注意力机制，进行自回归轨迹建模。
# PlanningDecoder（完整解码器）：包含多个 DecoderLayer，负责处理输入数据并输出轨迹预测。


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,#编码器输出（用于交叉注意力）
        tgt_key_padding_mask: Optional[Tensor] = None,#参考点的填充掩码（是否为有效点）
        memory_key_padding_mask: Optional[Tensor] = None,#编码器数据的填充掩码
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape
        
    # 【r2r_attn：参考点之间的自注意力】
        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt = torch.clamp(tgt, min=-1e6, max=1e6)
        tgt2 = self.norm1(tgt)
        # tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=1e6, neginf=-1e6)
        # print(f"tgt2 before r2r_attn: is {tgt2}")
        assert torch.isfinite(tgt2).all(), "NaN or Inf found in tgt2 before r2r_attn"
        # #   # Ensure key_padding_mask is not all zeros or ones
        # # tgt_key_padding_mask = tgt_key_padding_mask.repeat(M, 1)
        # # assert torch.isfinite(tgt_key_padding_mask).all(), "NaN or Inf found in tgt_key_padding_mask"
        
        # print(f"注意注意 tgt_key_padding_mask: {tgt_key_padding_mask}")
        # if torch.all(tgt_key_padding_mask == 0):
        #     print("Warning: tgt_key_padding_mask is all zeros.")
        # if torch.all(tgt_key_padding_mask == 1):
        #     print("Warning: tgt_key_padding_mask is all ones.")
        
        if torch.any(torch.all(tgt_key_padding_mask, dim=1)):
            print("Warning: The last column of tgt_key_padding_mask is all True.")
            return None 
        
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)     #自注意力（Self-Attention）
        )[0]
        # tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # print(f"注意注意 tgt2 after r2r_attn: is {tgt2}")
        
        assert torch.isfinite(tgt2).all(), "NaN or Inf found in tgt after r2r_attn"
        # Dropout + 残差连接（Residual Connection）防止梯度消失
        tgt = tgt + self.dropout1(tgt2)
        # tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e6, neginf=-1e6)
        assert torch.isfinite(tgt).all(), "NaN or Inf found in tgt after r2r_attn+dropout1"





        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt_valid = torch.clamp(tgt_valid, min=-1e6, max=1e6)
        tgt2_valid = self.norm2(tgt_valid)
        # tgt2_valid = torch.nan_to_num(tgt2_valid, nan=0.0, posinf=1e6, neginf=-1e6)
        assert torch.isfinite(tgt2_valid).all(), "NaN or Inf found in tgt2_valid after norm2"
        
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        # query = tgt2_valid + m_pos
        # key = tgt2_valid + m_pos
        # value = tgt2_valid
        # tgt2_valid = torch.nan_to_num(tgt2_valid, nan=0.0, posinf=1e6, neginf=-1e6)
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid


        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        
        
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]
        # tgt2= torch.nan_to_num(tgt2, nan=0.0, posinf=1e6, neginf=-1e6)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        
        # tgt= torch.nan_to_num(tgt, nan=0.0, posinf=1e6, neginf=-1e6)
        assert torch.isfinite(tgt).all(), "NaN or Inf found in tgt after attention block"
        
        tgt = tgt.reshape(bs, R, M, D)
        
        tgt = torch.clamp(tgt, min=-1e6, max=1e6)
        
        assert torch.isfinite(tgt).all(), "NaN or Inf found in tgt after clamping"

        return tgt


class PlanningDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        yaw_constraint=False,
        cat_x=False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))

        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)

    def forward(self, data, enc_data):
        enc_emb = enc_data["enc_emb"]
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]

        r_position = data["reference_line"]["position"]
        r_vector = data["reference_line"]["vector"]
        r_orientation = data["reference_line"]["orientation"]
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)


        # print(f"形状形状r_valid_mask shape: {r_valid_mask.shape}")
        
        
        # 形状形状r_valid_mask shape: torch.Size([32, 7, 120])
        # 形状形状r_valid_mask shape: torch.Size([32, 12, 120])
        # r_valid_mask.any(-1) 的返回值是一个形状为 [32, 7]
        #32个bs，
        #【bs , R , M 】（每隔1m一个点）
        # r_key_padding_mask的shape是【Bs R】
        
        r_feature = torch.cat(
            [
                r_position - r_position[..., 0:1, :2],
                r_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )

        bs, R, P, C = r_feature.shape   #【 bs  R  P点数120  C特征维度】
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)


        r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)
        m_emb = self.m_emb.repeat(bs, R, 1, 1)
        
        
        #q 是一个张量，它代表经过一系列变换和处理后的查询向量（query vector）

        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))
        
        # print(f"q after projection: min={q.min()}, max={q.max()}, mean={q.mean()}, std={q.std()}")
        
        for blk in self.decoder_blocks:
            q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
            )
            if q is None:  # 检测到 DecoderLayer 跳过该 batch
                print("Skipping batch in PlanningDecoder.")
                return None, None  # 直接返回 None，表示该 batch 无需训练
            
            
            q = torch.clamp(q, min=-30, max=45)#限制一下范围
            assert torch.isfinite(q).all()
            # print(f"q after blk: min={q.min()}, max={q.max()}, mean={q.mean()}, std={q.std()}")
            
        # print("一次一次一次一次forward结束！！！！#####################################################################")
            

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)
        

        # print(f"Original R: {data['reference_line']['position'].shape[1]}")
        # print(f"After r_valid_mask processing: {r_valid_mask.shape}")
        # print(f"After r_encoder: {r_emb.shape}")
        # print(f"Final q shape: {q.shape}")
        # print(f"Final pi shape: {pi.shape}")

        
        

        traj = torch.cat([loc, yaw, vel], dim=-1)
        
        # print(f"traj= {traj}  pi= {pi} ")
        return traj, pi
