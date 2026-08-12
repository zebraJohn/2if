import torch
import torch.nn as nn
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer

from model.Times2D import Times2DBackbone
from model.iTransformer import Model as iTransformerModel


class FreEnhancer(nn.Module):
    """② FreEformer 频域增强模块

    沿时间维做 FFT，在频域上用自注意力（实/虚双支，通道作为 token）学习
    特征分量间的关联，再 irfft 回时域，实现时-频融合增强。
    复用 FreEformer 的频域注意力思想 + 标准 FullAttention 层。
    输入/输出: [B, N, L]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.embed_size = getattr(configs, 'embed_size', 1)
        self.fp = int((self.seq_len + 1) / 2 + 0.5)          # 有效频点数
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.output_attention = configs.output_attention

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        d_in = self.fp * self.embed_size

        def _attn_layer():
            return EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation)

        # 实部支路
        self.proj_real = nn.Linear(d_in, self.d_model)
        self.encoder_real = Encoder(
            [_attn_layer() for _ in range(self.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.head_real = nn.Linear(self.d_model, self.fp)
        # 虚部支路
        self.proj_imag = nn.Linear(d_in, self.d_model)
        self.encoder_imag = Encoder(
            [_attn_layer() for _ in range(self.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.head_imag = nn.Linear(self.d_model, self.fp)

    def forward(self, z):  # z: [B, N, L]
        B, N, L = z.shape
        x = z.unsqueeze(-1) * self.embeddings                    # [B, N, L, D]
        xt = x.transpose(-1, -2)                                 # [B, N, D, L]
        x_fre = torch.fft.rfft(xt, dim=-1, norm='ortho')         # [B, N, D, fp]

        y_real, _ = self.encoder_real(self.proj_real(x_fre.real.flatten(-2)))  # [B, N, d_model]
        y_real = self.head_real(y_real).reshape(B, N, 1, self.fp)              # [B, N, 1, fp]
        y_imag, _ = self.encoder_imag(self.proj_imag(x_fre.imag.flatten(-2)))
        y_imag = self.head_imag(y_imag).reshape(B, N, 1, self.fp)

        y = torch.complex(y_real, y_imag)                        # [B, N, 1, fp]
        x_enh = torch.fft.irfft(y, n=L, dim=-1, norm='ortho')    # [B, N, 1, L]
        return x_enh.squeeze(-2)                                 # [B, N, L]


class DeviceAdapter(nn.Module):
    """③ 设备个性化 Adapter（窄两层 MLP），按 device_id 选择。

    use_adapter=0（本次实验单设备）：恒等映射，参数不参与前向；
    结构保留，后续多设备局放数据可直接启用。
    输入/输出: [B, N, L]（沿时间维 L 做逐设备非线性变换）
    """

    def __init__(self, in_features, n_devices=1, use_adapter=0):
        super().__init__()
        self.use_adapter = use_adapter
        self.n_devices = max(n_devices, 1)
        hidden = max(in_features // 2, 16)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, in_features),
            ) for _ in range(self.n_devices)
        ])

    def forward(self, z, device_id=0):
        if not self.use_adapter:
            return z
        return self.adapters[device_id](z)


class Model(nn.Module):
    """newidea 组合模型：Times2D → FreEformer → Adapter → iTransformer

    ① Times2D Backbone      多尺度周期折叠 + 二维卷积 + TSTiEncoder，输出周期特征 [B,C,L]
    ② FreEnhancer           频域自注意力时频增强，输出 [B,C,L]
    ③ DeviceAdapter         设备个性化（本次实验 identity），输出 [B,C,L]
    ④ iTransformer          倒置 token 自注意力提纯 + 线性投影，输出 [B,pred_len,C]

    外部 RevIN 归一化：输入按 (mean,std) 归一化，输出再反归一化回原始尺度，
    保证 MSE/MAE 在原始数据尺度上可比。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # ① Times2D 多尺度周期特征提取
        self.t2d = Times2DBackbone(configs)
        # ② FreEformer 频域增强
        self.fre = FreEnhancer(configs)
        # ③ 设备 Adapter（预埋，本实验 identity）
        self.adapter = DeviceAdapter(in_features=configs.seq_len,
                                     n_devices=getattr(configs, 'n_devices', 1),
                                     use_adapter=getattr(configs, 'use_adapter', 0))
        # ④ iTransformer 提纯预测（内部 use_norm 关闭，由外部 RevIN 统一归一化）
        use_norm_ori = getattr(configs, 'use_norm', 1)
        configs.use_norm = 0
        self.itrans = iTransformerModel(configs)
        configs.use_norm = use_norm_ori

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_y=None):
        # batch_x: [B, L, C]
        B, L, C = batch_x.shape
        # 外部 RevIN：保证输出回到原始数据尺度
        means = batch_x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (batch_x - means) / stdev

        x = x_norm.permute(0, 2, 1)                 # [B, C, L]
        z_t2d = self.t2d.forward_features(x)        # [B, C, L]
        z_fre = self.fre(z_t2d)                     # [B, C, L]
        z_adapt = self.adapter(z_fre)               # [B, C, L]（identity）
        z_in = z_adapt.permute(0, 2, 1)             # [B, L, C]

        out = self.itrans(z_in, None, None, None)   # [B, pred_len, C]
        if isinstance(out, tuple):                  # output_attention=True 时
            out = out[0]
        out = out * stdev + means                   # denorm
        return out
