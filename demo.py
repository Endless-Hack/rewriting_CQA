import torch
import torch.nn as nn

transformer_model = nn.Transformer(nhead=8, num_encoder_layers=2)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
print(out.shape)


encoder_layer = nn.TransformerEncoderLayer(d_model=400, nhead=8, dim_feedforward=8192)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
src = torch.rand(10, 400)
out = transformer_encoder(src)
print(out.shape)

out = torch.mean(out, dim=1)
print(out.shape)

# 加一个全连接网络 降维到query_embedding size
