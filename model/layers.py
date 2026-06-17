# import  torch
# from torchvision.models import resnet18
# import torch.nn.functional as F
# import numpy as np
# import math
# import torch.nn as nn
# from torch.autograd import Function
# class EmbeddingLayer(torch.nn.Module):

#     def __init__(self, field_dims, embed_dim):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#         torch.nn.init.xavier_uniform_(self.embedding.weight.data)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         x = x + x.new_tensor(self.offsets).unsqueeze(0)
#         return self.embedding(x)

# class MultiLayerPerceptron(torch.nn.Module):

#     def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
#         super().__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim, embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         if output_layer:
#             layers.append(torch.nn.Linear(input_dim, 1))
#         self.mlp = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, embed_dim)``
#         """
#         return self.mlp(x)

# class MLP(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,1))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)

# class MLP_Mu(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP_Mu, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,9))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)

# class MLP_fusion(torch.nn.Module):
#     def __init__(self,input_dim,out_dim,embed_dims,dropout):
#         super(MLP_fusion, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,out_dim))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)
# """
# class MLP_fusion_gate(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP_fusion_gate, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,768))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)
# """
# class clip_fuion(torch.nn.Module):
#     def __init__(self,input_dim,out_dim,embed_dims,dropout):
#         super(clip_fuion, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,out_dim))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)



# class cnn_extractor(torch.nn.Module):
#     def __init__(self,input_size,feature_kernel):
#         super(cnn_extractor, self).__init__()
#         self.convs =torch.nn.ModuleList(
#             [torch.nn.Conv1d(input_size,feature_num,kernel)
#             for kernel,feature_num in feature_kernel.items()]
#         )
#     def forward(self,input_data):
#         input_data = input_data.permute(0, 2, 1)
#         feature = [conv(input_data)for conv in self.convs]
#         feature = [torch.max_pool1d(f,f.shape[-1])for f in feature]
#         feature = torch.cat(feature,dim = 1)
#         feature = feature.view([-1,feature.shape[1]])
#         return feature

# class image_cnn_extractor(nn.Module):
#     def __init__(self):
#         super(image_cnn_extractor, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

#         # Pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Fully connected layers
#         self.fc1 = nn.Linear(256 * 24 * 96, 512)
#         self.fc2 = nn.Linear(512, 320)

#         # Activation function
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Convolutional layers with ReLU activation and pooling
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)

#         # Flatten the output for fully connected layers
#         x = x.view(-1, 256 * 24 * 96)

#         # Fully connected layers with ReLU activation
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x

# class image_extractor(torch.nn.Module):
#     def __init__(self,out_channels):
#         super(image_extractor, self).__init__()
#         self.img_backbone = resnet18(pretrained=True)
#         self.img_model = torch.nn.ModuleList([
#             self.img_backbone.conv1,
#             self.img_backbone.bn1,
#             self.img_backbone.relu,
#             self.img_backbone.layer1,
#             self.img_backbone.layer2,
#             self.img_backbone.layer3,
#             self.img_backbone.layer4
#         ])
#         self.img_model = torch.nn.Sequential(*self.img_model)
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)
#     def forward(self,img):
#         n_batch = img.size(0)
#         img_out = self.img_model(img)
#         img_out = self.avg_pool(img_out)#([64, 512, 1, 1])
#         img_out = img_out.view(n_batch, -1)#([64, 512])
#         img_out = self.img_fc(img_out)#([64, 320])
#         img_out = F.normalize(img_out, p=2, dim=-1)
#         return img_out

# class classifier(torch.nn.Module):
#     def __init__(self,out_dim=1):
#         super(classifier, self).__init__()
#         self.trim = nn.Sequential(
#             nn.Linear(self.unified_dim, 64),
#             nn.SiLU(),
#             # SimpleGate(),
#             # nn.BatchNorm1d(64),
#             # nn.Dropout(0.2),
#         )
#         self.classifier1 = nn.Sequential(
#             nn.Linear(64, out_dim),
#         )
#     def forward(self,x):
#         x = self.classifier1(self.trim(x))
#         return x

# class MaskAttention(torch.nn.Module):
#     def __init__(self,input_dim):
#         super(MaskAttention, self).__init__()
#         self.Line = torch.nn.Linear(input_dim,1)

#     def forward(self,input,mask):
#         score = self.Line(input).view(-1,input.size(1))
#         if mask is not None:
#             score = score.masked_fill(mask == 0, float("-inf"))
#         score = torch.softmax(score, dim=-1).unsqueeze(1)
#         output = torch.matmul(score,input).squeeze(1)
#         return output

# class TokenAttention(torch.nn.Module):
#     """
#     Compute attention layer
#     """

#     def __init__(self, input_shape):
#         super(TokenAttention, self).__init__()
#         self.attention_layer = nn.Sequential(
#                             torch.nn.Linear(input_shape, input_shape),
#                             nn.SiLU(),
#                             #SimpleGate(dim=2),
#                             torch.nn.Linear(input_shape, 1),
#         )

#     def forward(self, inputs):
#         scores = self.attention_layer(inputs).view(-1, inputs.size(1))
#         #scores = torch.softmax(scores, dim=-1).unsqueeze(1)
#         scores = scores.unsqueeze(1)
#         outputs = torch.matmul(scores, inputs).squeeze(1)
#         # scores = self.attention_layer(inputs)
#         # outputs = scores*inputs
#         return outputs, scores
# class Attention(torch.nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """

#     def forward(self, query, key, value, mask=None, dropout=None):
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(query.size(-1))

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float("-inf"))

#         p_attn = F.softmax(scores, dim=-1)

#         if dropout is not None:
#             p_attn = dropout(p_attn)

#         return torch.matmul(p_attn, value), p_attn
# class MultiHeadedAttention(torch.nn.Module):
#     """
#     Take in model size and number of heads.
#     """

#     def __init__(self, h, d_model, dropout=0.1):
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0

#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h

#         self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
#         self.output_linear = torch.nn.Linear(d_model, d_model)
#         self.attention = Attention()

#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#         if mask is not None:
#             mask = mask.repeat(1, self.h, 1, 1)
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]

#         # 2) Apply attention on all the projected vectors in batch.
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

#         return self.output_linear(x), attn
# class Resnet(torch.nn.Module):
#     def __init__(self,out_channels):
#         super(Resnet, self).__init__()
#         self.img_backbone = resnet18(pretrained=True)
#         self.img_model = torch.nn.ModuleList([
#             self.img_backbone.conv1,
#             self.img_backbone.bn1,
#             self.img_backbone.relu,
#             self.img_backbone.layer1,
#             self.img_backbone.layer2,
#             self.img_backbone.layer3,
#             self.img_backbone.layer4
#         ])
#         self.img_model = torch.nn.Sequential(*self.img_model)
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

#     def forward(self, img):
#         n_batch = img.size(0)
#         img_out = self.img_model(img)
#         img_out = self.avg_pool(img_out)
#         img_out = img_out.view(n_batch, -1)
#         img_out = self.img_fc(img_out)
#         img_out = F.normalize(img_out, p=2, dim=-1)
#         return img_out

# class ReverseLayerF(Function):
#     @staticmethod
#     def forward(ctx, input_, alpha):
#         ctx.alpha = alpha
#         return input_

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None


# 原版
# import  torch
# from torchvision.models import resnet18
# import torch.nn.functional as F
# import numpy as np
# import math
# import torch.nn as nn
# from torch.autograd import Function
# class EmbeddingLayer(torch.nn.Module):

#     def __init__(self, field_dims, embed_dim):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#         torch.nn.init.xavier_uniform_(self.embedding.weight.data)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         x = x + x.new_tensor(self.offsets).unsqueeze(0)
#         return self.embedding(x)

# class MultiLayerPerceptron(torch.nn.Module):

#     def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
#         super().__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim, embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         if output_layer:
#             layers.append(torch.nn.Linear(input_dim, 1))
#         self.mlp = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, embed_dim)``
#         """
#         return self.mlp(x)

# class MLP(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,1))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)

# class MLP_Mu(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP_Mu, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,9))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)

# class MLP_fusion(torch.nn.Module):
#     def __init__(self,input_dim,out_dim,embed_dims,dropout):
#         super(MLP_fusion, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,out_dim))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)
# """
# class MLP_fusion_gate(torch.nn.Module):
#     def __init__(self,input_dim,embed_dims,dropout):
#         super(MLP_fusion_gate, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,768))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)
# """
# class clip_fuion(torch.nn.Module):
#     def __init__(self,input_dim,out_dim,embed_dims,dropout):
#         super(clip_fuion, self).__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim,embed_dim))
#             layers.append(torch.nn.BatchNorm1d(embed_dim))
#             layers.append(torch.nn.GELU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         layers.append(torch.nn.Linear(input_dim,out_dim))
#         self.mlp = torch.nn.Sequential(*layers)
#     def forward(self,x):
#         return self.mlp(x)



# class cnn_extractor(torch.nn.Module):
#     def __init__(self,input_size,feature_kernel):
#         super(cnn_extractor, self).__init__()
#         self.convs =torch.nn.ModuleList(
#             [torch.nn.Conv1d(input_size,feature_num,kernel)
#             for kernel,feature_num in feature_kernel.items()]
#         )
#     def forward(self,input_data):
#         input_data = input_data.permute(0, 2, 1)
#         feature = [conv(input_data)for conv in self.convs]
#         feature = [torch.max_pool1d(f,f.shape[-1])for f in feature]
#         feature = torch.cat(feature,dim = 1)
#         feature = feature.view([-1,feature.shape[1]])
#         return feature

# class image_cnn_extractor(nn.Module):
#     def __init__(self):
#         super(image_cnn_extractor, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

#         # Pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Fully connected layers
#         self.fc1 = nn.Linear(256 * 24 * 96, 512)
#         self.fc2 = nn.Linear(512, 320)

#         # Activation function
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Convolutional layers with ReLU activation and pooling
#         x = self.relu(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool(x)

#         # Flatten the output for fully connected layers
#         x = x.view(-1, 256 * 24 * 96)

#         # Fully connected layers with ReLU activation
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x

# class image_extractor(torch.nn.Module):
#     def __init__(self,out_channels):
#         super(image_extractor, self).__init__()
#         self.img_backbone = resnet18(pretrained=True)
#         self.img_model = torch.nn.ModuleList([
#             self.img_backbone.conv1,
#             self.img_backbone.bn1,
#             self.img_backbone.relu,
#             self.img_backbone.layer1,
#             self.img_backbone.layer2,
#             self.img_backbone.layer3,
#             self.img_backbone.layer4
#         ])
#         self.img_model = torch.nn.Sequential(*self.img_model)
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)
#     def forward(self,img):
#         n_batch = img.size(0)
#         img_out = self.img_model(img)
#         img_out = self.avg_pool(img_out)#([64, 512, 1, 1])
#         img_out = img_out.view(n_batch, -1)#([64, 512])
#         img_out = self.img_fc(img_out)#([64, 320])
#         img_out = F.normalize(img_out, p=2, dim=-1)
#         return img_out

# class classifier(torch.nn.Module):
#     def __init__(self,out_dim=1):
#         super(classifier, self).__init__()
#         self.trim = nn.Sequential(
#             nn.Linear(self.unified_dim, 64),
#             nn.SiLU(),
#             # SimpleGate(),
#             # nn.BatchNorm1d(64),
#             # nn.Dropout(0.2),
#         )
#         self.classifier1 = nn.Sequential(
#             nn.Linear(64, out_dim),
#         )
#     def forward(self,x):
#         x = self.classifier1(self.trim(x))
#         return x

# class MaskAttention(torch.nn.Module):
#     def __init__(self,input_dim):
#         super(MaskAttention, self).__init__()
#         self.Line = torch.nn.Linear(input_dim,1)

#     def forward(self,input,mask):
#         score = self.Line(input).view(-1,input.size(1))
#         if mask is not None:
#             score = score.masked_fill(mask == 0, float("-inf"))
#         score = torch.softmax(score, dim=-1).unsqueeze(1)
#         output = torch.matmul(score,input).squeeze(1)
#         return output

# class TokenAttention(torch.nn.Module):
#     """
#     Compute attention layer
#     """

#     def __init__(self, input_shape):
#         super(TokenAttention, self).__init__()
#         self.attention_layer = nn.Sequential(
#                             torch.nn.Linear(input_shape, input_shape),
#                             nn.SiLU(),
#                             #SimpleGate(dim=2),
#                             torch.nn.Linear(input_shape, 1),
#         )

#     def forward(self, inputs):
#         scores = self.attention_layer(inputs).view(-1, inputs.size(1))
#         #scores = torch.softmax(scores, dim=-1).unsqueeze(1)
#         scores = scores.unsqueeze(1)
#         outputs = torch.matmul(scores, inputs).squeeze(1)
#         # scores = self.attention_layer(inputs)
#         # outputs = scores*inputs
#         return outputs, scores
# class Attention(torch.nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """

#     def forward(self, query, key, value, mask=None, dropout=None):
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(query.size(-1))

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float("-inf"))

#         p_attn = F.softmax(scores, dim=-1)

#         if dropout is not None:
#             p_attn = dropout(p_attn)

#         return torch.matmul(p_attn, value), p_attn
# class MultiHeadedAttention(torch.nn.Module):
#     """
#     Take in model size and number of heads.
#     """

#     def __init__(self, h, d_model, dropout=0.1):
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0

#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h

#         self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
#         self.output_linear = torch.nn.Linear(d_model, d_model)
#         self.attention = Attention()

#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#         if mask is not None:
#             mask = mask.repeat(1, self.h, 1, 1)
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]

#         # 2) Apply attention on all the projected vectors in batch.
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

#         return self.output_linear(x), attn
# class Resnet(torch.nn.Module):
#     def __init__(self,out_channels):
#         super(Resnet, self).__init__()
#         self.img_backbone = resnet18(pretrained=True)
#         self.img_model = torch.nn.ModuleList([
#             self.img_backbone.conv1,
#             self.img_backbone.bn1,
#             self.img_backbone.relu,
#             self.img_backbone.layer1,
#             self.img_backbone.layer2,
#             self.img_backbone.layer3,
#             self.img_backbone.layer4
#         ])
#         self.img_model = torch.nn.Sequential(*self.img_model)
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

#     def forward(self, img):
#         n_batch = img.size(0)
#         img_out = self.img_model(img)
#         img_out = self.avg_pool(img_out)
#         img_out = img_out.view(n_batch, -1)
#         img_out = self.img_fc(img_out)
#         img_out = F.normalize(img_out, p=2, dim=-1)
#         return img_out

# class ReverseLayerF(Function):
#     @staticmethod
#     def forward(ctx, input_, alpha):
#         ctx.alpha = alpha
#         return input_

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None


import torch
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Function


def masked_softmax(scores, mask=None, dim=-1, mask_fill_value=-1e4):
    if mask is None:
        return torch.softmax(scores, dim=dim)
    if mask.dim() < scores.dim():
        for _ in range(scores.dim() - mask.dim()):
            mask = mask.unsqueeze(1)
    mask = mask.to(dtype=scores.dtype)
    scores = scores.masked_fill(mask == 0, mask_fill_value)
    mask_sum = mask.sum(dim=dim, keepdim=True)
    scores = torch.where(mask_sum > 0, scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=dim)
    probs = torch.where(mask_sum > 0, probs, torch.zeros_like(probs))
    return probs


# ------------------------------
# 新增 LayerNorm 类（解决导入错误）
# ------------------------------
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization），支持高维输入"""
    def __init__(self, normalized_shape, eps=1e-12, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape  # 归一化的维度（如 [320]）
        self.eps = eps  # 防止除零的小值
        self.elementwise_affine = elementwise_affine  # 是否使用可学习的缩放和平移参数
        
        # 定义可学习参数（若需要）
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(normalized_shape))  # 缩放参数
            self.bias = nn.Parameter(torch.Tensor(normalized_shape))    # 平移参数
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """初始化可学习参数"""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 缩放参数初始化为 1
            nn.init.zeros_(self.bias)   # 平移参数初始化为 0

    def forward(self, x):
        """前向传播：对输入进行层归一化"""
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


# ------------------------------
# 以下为原有类（保持不变）
# ------------------------------
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_Mu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP_Mu, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 9))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class clip_fuion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(clip_fuion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnn_extractor(torch.nn.Module):
    def __init__(self, input_size, feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()]
        )

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1)
        feature = [conv(input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class image_cnn_extractor(nn.Module):
    def __init__(self):
        super(image_cnn_extractor, self).__init__()
        self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 24 * 96, 512)
        self.fc2 = nn.Linear(512, 320)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 24 * 96)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class image_extractor(torch.nn.Module):
    def __init__(self, out_channels):
        super(image_extractor, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out


class classifier(torch.nn.Module):
    def __init__(self, out_dim=1):
        super(classifier, self).__init__()
        self.trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.classifier1(self.trim(x))
        return x


class MaskAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim, 1)

    def forward(self, input, mask):
        score = self.Line(input).view(-1, input.size(1))
        score = masked_softmax(score, mask=mask, dim=-1).unsqueeze(1)
        output = torch.matmul(score, input).squeeze(1)
        return output


class TokenAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class CrossModalCalibration(nn.Module):
    def __init__(self, local_dim, global_dim, calib_dim, dropout=0.1):
        super().__init__()
        self.local_dim = local_dim
        self.calib_dim = calib_dim
        self.text_global_to_local = nn.Linear(global_dim, local_dim)
        self.image_global_to_local = nn.Linear(global_dim, local_dim)
        self.text_local_proj = nn.Linear(local_dim, calib_dim)
        self.image_local_proj = nn.Linear(local_dim, calib_dim)
        self.text_global_proj = nn.Linear(global_dim, calib_dim)
        self.image_global_proj = nn.Linear(global_dim, calib_dim)
        self.text_out_proj = nn.Linear(calib_dim, local_dim)
        self.image_out_proj = nn.Linear(calib_dim, local_dim)
        self.norm = nn.LayerNorm(calib_dim)
        self.dropout = nn.Dropout(dropout)

    def _self_calibrate(self, local_feat, global_feat, global_to_local, mask=None):
        global_local = global_to_local(global_feat).unsqueeze(1)
        scores = (local_feat * global_local).sum(-1) / math.sqrt(self.local_dim)
        weights = masked_softmax(scores, mask=mask, dim=1).unsqueeze(-1)
        return local_feat * weights

    def forward(self, text_local, image_local, text_global, image_global, text_mask=None, image_mask=None):
        text_self = self._self_calibrate(text_local, text_global, self.text_global_to_local, text_mask)
        image_self = self._self_calibrate(image_local, image_global, self.image_global_to_local, image_mask)

        text_c = self.norm(self.text_local_proj(text_self) + self.text_global_proj(text_global).unsqueeze(1))
        image_c = self.norm(self.image_local_proj(image_self) + self.image_global_proj(image_global).unsqueeze(1))

        attn_logits = torch.matmul(text_c, image_c.transpose(1, 2)) / math.sqrt(self.calib_dim)
        if image_mask is not None:
            attn = masked_softmax(attn_logits, mask=image_mask.unsqueeze(1), dim=-1)
        else:
            attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        text_cc = torch.matmul(attn, image_c) + text_c
        image_cc = torch.matmul(attn.transpose(1, 2), text_c) + image_c

        text_out = self.text_out_proj(text_cc) + text_local
        image_out = self.image_out_proj(image_cc) + image_local

        if text_mask is not None:
            text_out = text_out * text_mask.unsqueeze(-1)
        if image_mask is not None:
            image_out = image_out * image_mask.unsqueeze(-1)
        return text_out, image_out


class HierarchicalConsistencyVerifier(nn.Module):
    def __init__(self, local_dim, global_dim, alpha=0.5, temperature=0.1, tau_init=0.0):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.alpha = alpha
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.eps = 1e-6

    def forward(self, text_local, image_local, text_global, image_global,
                text_mask=None, image_mask=None, labels=None):
        batch_size = text_local.size(0)
        text_len = text_local.size(1)
        image_len = image_local.size(1)

        text_local_norm = F.normalize(text_local, p=2, dim=-1)
        image_local_norm = F.normalize(image_local, p=2, dim=-1)
        sim = torch.matmul(text_local_norm, image_local_norm.transpose(1, 2))

        if text_mask is not None:
            sim = sim.masked_fill(text_mask.unsqueeze(2) == 0, -1e4)
        if image_mask is not None:
            sim = sim.masked_fill(image_mask.unsqueeze(1) == 0, -1e4)

        s_l_max = torch.amax(sim, dim=(1, 2))

        sim_avg = sim.clone()
        if text_mask is not None:
            sim_avg = sim_avg.masked_fill(text_mask.unsqueeze(2) == 0, 0.0)
        if image_mask is not None:
            sim_avg = sim_avg.masked_fill(image_mask.unsqueeze(1) == 0, 0.0)

        if text_mask is None:
            text_count = torch.full((batch_size, 1), text_len, device=text_local.device)
        else:
            text_count = text_mask.sum(dim=1, keepdim=True)
        if image_mask is None:
            image_count = torch.full((batch_size, 1), image_len, device=image_local.device)
        else:
            image_count = image_mask.sum(dim=1, keepdim=True)
        denom = (text_count * image_count).clamp_min(1.0)
        s_l_avg = sim_avg.sum(dim=(1, 2)) / denom.squeeze(1)

        s_l = self.alpha * s_l_max + (1.0 - self.alpha) * s_l_avg
        if text_mask is not None or image_mask is not None:
            text_valid = text_mask.sum(dim=1) > 0 if text_mask is not None else torch.ones_like(s_l, dtype=torch.bool)
            image_valid = image_mask.sum(dim=1) > 0 if image_mask is not None else torch.ones_like(s_l, dtype=torch.bool)
            valid = text_valid & image_valid
            s_l = torch.where(valid, s_l, torch.zeros_like(s_l))

        text_global_norm = F.normalize(text_global, p=2, dim=-1)
        image_global_norm = F.normalize(image_global, p=2, dim=-1)
        s_g0 = (text_global_norm * image_global_norm).sum(dim=-1)
        sigma = torch.exp(self.log_sigma).clamp_min(1e-3)
        s_g = torch.sigmoid((s_g0 - self.tau) / sigma)

        cons_loss = None
        if labels is not None:
            labels = labels.float().view(-1)
            real_mask = 1.0 - labels
            fake_mask = labels
            loss_real = real_mask * ((1.0 - s_l) ** 2 + (1.0 - s_g) ** 2)
            loss_fake = fake_mask * torch.min(s_l ** 2, s_g ** 2)
            cons_loss = (loss_real + loss_fake).mean()

        return s_l, s_g, cons_loss


class HierarchicalFusionNoiseFilter(nn.Module):
    def __init__(self, local_dim, global_dim, fusion_dim=160):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.fusion_dim = fusion_dim
        self.text_local_proj = nn.Linear(local_dim, fusion_dim)
        self.image_local_proj = nn.Linear(local_dim, fusion_dim)
        self.text_global_proj = nn.Linear(global_dim, fusion_dim)
        self.image_global_proj = nn.Linear(global_dim, fusion_dim)
        self.noise_filter = nn.Linear(2 * fusion_dim, 2 * fusion_dim)
        self.gate = nn.Parameter(torch.ones(2 * fusion_dim))

    def forward(self, text_local, image_local, text_global, image_global,
                s_l, s_g, text_mask=None, image_mask=None):
        text_l = self.text_local_proj(text_local)
        image_l = self.image_local_proj(image_local)
        local_seq = torch.cat([text_l, image_l], dim=1)

        text_g = self.text_global_proj(text_global).unsqueeze(1)
        image_g = self.image_global_proj(image_global).unsqueeze(1)
        global_seq = torch.cat([text_g, image_g], dim=1)

        w_l = torch.sigmoid(s_l).unsqueeze(-1).unsqueeze(-1)
        w_g = torch.sigmoid(s_g).unsqueeze(-1).unsqueeze(-1)
        local_seq = local_seq * w_l
        global_seq = global_seq * w_g

        attn_logits = torch.matmul(local_seq, global_seq.transpose(1, 2)) / math.sqrt(self.fusion_dim)
        attn = torch.softmax(attn_logits, dim=-1)
        fused_local = torch.matmul(attn, global_seq) + local_seq

        if text_mask is None:
            text_mask = torch.ones(text_local.size(0), text_local.size(1), device=text_local.device)
        if image_mask is None:
            image_mask = torch.ones(image_local.size(0), image_local.size(1), device=image_local.device)
        local_mask = torch.cat([text_mask, image_mask], dim=1).unsqueeze(-1)
        pooled_local = (fused_local * local_mask).sum(dim=1) / local_mask.sum(dim=1).clamp_min(1.0)
        pooled_global = global_seq.mean(dim=1)

        fused_init = torch.cat([pooled_local, pooled_global], dim=-1)
        filtered = F.relu(self.noise_filter(fused_init))
        fused_final = filtered * self.gate + fused_init
        return fused_final


class Attention(torch.nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        p_attn = masked_softmax(scores, mask=mask, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x), attn


class Resnet(torch.nn.Module):
    def __init__(self, out_channels):
        super(Resnet, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None