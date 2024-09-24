import torch
import torch.nn as nn
from torchvision.models import vgg16


# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.query_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features, queries):
        B, C, H, W = features.shape
        features = features.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        queries = queries.expand(B, -1, -1)  # [B, 1, C]

        # Cross-attention between queries and features
        attn_output, _ = self.multihead_attn(queries, features, features)
        output = self.norm(attn_output + queries)
        return output


# VGG Backbone with Cross-Attention Decoder
class VGG_CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_classes=3):
        super(VGG_CrossAttentionDecoder, self).__init__()
        vgg = vgg16(pretrained=True)
        # 提取VGG16的不同层次特征
        self.backbone1 = nn.Sequential(*list(vgg.features.children())[:10])  # 第1层输出 (batch_size, 128, 56, 56)
        self.backbone2 = nn.Sequential(*list(vgg.features.children())[10:17])  # 第2层输出 (batch_size, 256, 28, 28)
        self.backbone3 = nn.Sequential(*list(vgg.features.children())[17:])  # 第3层输出 (batch_size, 512, 14, 14)

        # Cross-Attention blocks
        self.cross_att1 = CrossAttention(embed_dim, num_heads)
        self.cross_att2 = CrossAttention(embed_dim, num_heads)

        # 修改为128通道输入
        self.conv1x1_low = nn.Conv2d(128, embed_dim, kernel_size=1)  # 将输入通道数调整为128
        self.conv1x1_high = nn.Conv2d(512, embed_dim, kernel_size=1)

        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # 获取多尺度特征
        low_features = self.backbone1(x)  # (batch_size, 128, 56, 56)
        mid_features = self.backbone2(low_features)  # (batch_size, 256, 28, 28)
        high_features = self.backbone3(mid_features)  # (batch_size, 512, 14, 14)

        # 1x1卷积调整维度
        low_features = self.conv1x1_low(low_features)  # (batch_size, 256, 56, 56)
        high_features = self.conv1x1_high(high_features)  # (batch_size, 256, 14, 14)

        # Cross-Attention融合
        query_embed = torch.randn(1, low_features.size(1), device=x.device)  # 创建查询向量
        attn_low = self.cross_att1(low_features, query_embed)  # (batch_size, 1, embed_dim)
        attn_high = self.cross_att2(high_features, attn_low)  # (batch_size, 1, embed_dim)

        # 不用view，使用unsqueeze和permute调整维度
        attn_high = attn_high.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, embed_dim, 1, 1)

        high_up = torch.nn.functional.interpolate(attn_high, size=(112, 112), mode='bilinear',
                                                  align_corners=False)  # 上采样到112x112

        low_up = torch.nn.functional.interpolate(low_features, size=(112, 112), mode='bilinear',
                                                 align_corners=False)  # 上采样到112x112

        # 最终的掩码预测
        mask_pred = high_up + low_up  # 特征融合
        mask_pred = self.final_conv(mask_pred)  # 输出分割掩码

        # 上采样到原始尺寸 (224, 224)，以确保与标签的分辨率一致
        mask_pred = torch.nn.functional.interpolate(mask_pred, size=(224, 224), mode='bilinear', align_corners=False)
        mask_pred = torch.sigmoid(mask_pred)

        return mask_pred
