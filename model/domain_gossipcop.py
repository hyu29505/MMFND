# model/domain.py

import logging
logger = logging.getLogger(__name__)

import os
import tqdm
import numpy as np
import torch
from transformers import BertModel, CLIPModel
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 MAE 模型定义 (models_mae.py 在根目录)
try:
    import models_mae
except ImportError:
    logger.error("无法导入 models_mae。请确保 models_mae.py 文件在项目根目录。")
    raise

# 尝试导入 utils 函数 (utils.py 在 utils/ 子目录)
try:
    from utils.utils_gossipcop import clipdata2gpu, Averager, calculate_metrics, Recorder
except ImportError as e:
    logger.error(f"无法从 utils.utils 导入函数: {e}. 请确保 utils/utils.py 存在。")
    raise

# 尝试导入自定义层和 timm Block (layers.py, pivot.py 在 model/ 目录)
# 注意：如果这些文件不存在或有错误，你需要提供它们的正确实现或占位符
try:
    from .layers import * # 假设这里会成功导入您提供的 layers.py
    from .pivot import *
except ImportError:
    logger.warning("无法导入自定义的 .layers 或 .pivot。如果模型依赖它们，将会出错。下面将使用占位符定义。")
    # 定义必要的占位符 (如果.layers导入失败，这些将被使用)
    class MaskAttention(nn.Module):
        def __init__(self, dim): super().__init__(); self.dim = dim
        def forward(self, feat, mask):
            if feat is None or mask is None: return torch.zeros(1, self.dim) if feat is None else torch.zeros(feat.shape[0], self.dim, device=feat.device)
            if mask.dim() == 2 and feat.dim() == 3: mask = mask.unsqueeze(-1)
            if mask.shape != feat.shape:
                if mask.shape[-1] == 1 and feat.shape[-1] == self.dim: mask = mask.expand_as(feat)
                else: logger.warning(f"Mask shape {mask.shape} mismatch {feat.shape}. Using mean pooling."); return torch.mean(feat, dim=1)
            masked_feat = feat * mask
            sum_masked_feat = torch.sum(masked_feat, dim=1); sum_mask = torch.sum(mask, dim=1)
            return sum_masked_feat / (sum_mask + 1e-9)

    class TokenAttention(nn.Module):
        def __init__(self, dim):
            super().__init__(); self.dim = dim; self.attention_weights = nn.Linear(dim, 1)
        def forward(self, feat):
            if feat is None: return torch.zeros(1, self.dim), None
            if feat.dim() != 3 or feat.shape[2] != self.dim : logger.error(f"TokenAttention shape error. Expected (B, L, {self.dim}), got {feat.shape}"); return torch.zeros(feat.shape[0], self.dim, device=feat.device), None
            e = self.attention_weights(feat); alpha = torch.softmax(e, dim=1); context = torch.bmm(alpha.transpose(1, 2), feat).squeeze(1); return context, alpha

    class MLP_fusion(nn.Module):
        def __init__(self, in_dim, out_dim, hidden_dims_list, dropout_rate):
            super().__init__(); layers = []; current_dim = in_dim
            if not hidden_dims_list: layers.append(nn.Linear(current_dim, out_dim))
            else:
                for h_dim in hidden_dims_list: layers.append(nn.Linear(current_dim, h_dim)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout_rate)); current_dim = h_dim
                layers.append(nn.Linear(current_dim, out_dim))
            self.network = nn.Sequential(*layers)
        def forward(self, x): return self.network(x)

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden_dims_list, dropout_rate):
            super().__init__(); layers = []; current_dim = in_dim
            if not hidden_dims_list: layers.append(nn.Linear(current_dim, 1))
            else:
                for h_dim in hidden_dims_list: layers.append(nn.Linear(current_dim, h_dim)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout_rate)); current_dim = h_dim
                layers.append(nn.Linear(current_dim, 1))
            self.network = nn.Sequential(*layers)
        def forward(self, x): return self.network(x)

    class cnn_extractor(nn.Module): # 这是占位符 cnn_extractor
        def __init__(self, in_dim_seq, feature_kernel_unused, out_dim=320):
            super().__init__()
            self.out_dim = out_dim
            self.in_dim = in_dim_seq
            self.pool_and_reduce = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim),
                nn.ReLU()
            )
        def forward(self, x_seq):
            if x_seq is None: return torch.zeros(1, self.out_dim, device=x_seq.device if hasattr(x_seq, 'device') else 'cpu')
            if x_seq.shape[-1] != self.in_dim:
                logger.warning(f"Placeholder cnn_extractor input feature dim mismatch. Expected {self.in_dim}, got {x_seq.shape[-1]}. Returning zeros.")
                return torch.zeros(x_seq.shape[0], self.out_dim, device=x_seq.device)
            if x_seq.dim() == 3:
                pooled_x = torch.mean(x_seq, dim=1)
            elif x_seq.dim() == 2:
                pooled_x = x_seq
            else:
                logger.error(f"Placeholder cnn_extractor input shape error: {x_seq.shape}, returning zeros.");
                return torch.zeros(x_seq.shape[0] if x_seq.dim()>0 else 1, self.out_dim, device=x_seq.device)
            output = self.pool_and_reduce(pooled_x)
            if output.shape[-1] != self.out_dim:
                logger.error(f"Placeholder cnn_extractor output shape error. Expected (*, {self.out_dim}), got {output.shape}. Check Linear layer.")
                return torch.zeros(output.shape[0], self.out_dim, device=output.device)
            return output

    class LayerNorm(nn.Module):
        def __init__(self, dim, eps=1e-12): super().__init__(); self.norm = nn.LayerNorm(dim, eps=eps)
        def forward(self, x): return self.norm(x) if x is not None else None

    class TransformerLayer(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()
        def forward(self, x, mask=None): return self.fc(x)

    class MLP_trans(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()
        def forward(self, x): return self.fc(x)


from timm.models.vision_transformer import Block

class SimpleGate(nn.Module): # 从 layers.py 移动到这里或确保 layers.py 中有定义（如果占位符未被使用）
    def __init__(self, dim=1): super(SimpleGate, self).__init__(); self.dim = dim
    def forward(self, x): x1, x2 = x.chunk(2, dim=self.dim); return x1 * x2

class AdaIN(nn.Module): # 从 layers.py 移动到这里或确保 layers.py 中有定义
    def __init__(self): super().__init__()
    def mu(self, x):
        if x is None: return None
        if x.dim() == 3: return torch.mean(x, dim=1)
        elif x.dim() == 2: return torch.mean(x, dim=0, keepdim=True)
        else: return torch.mean(x)

    def sigma(self, x):
        if x is None: return None
        if x.dim() == 3:
            mu_val = self.mu(x).unsqueeze(1)
            return torch.sqrt(torch.mean((x - mu_val)**2, dim=1) + 1e-8)
        elif x.dim() == 2:
            return torch.sqrt(torch.mean((x - self.mu(x))**2, dim=0, keepdim=True) + 1e-8)
        else: return torch.std(x) + 1e-8

    def forward(self, x, mu, sigma):
        if x is None or mu is None or sigma is None: return x
        x_dim = x.dim()
        x_mean = self.mu(x)
        x_std = self.sigma(x)

        if x_dim == 3:
            if x_mean.dim() == 2: x_mean = x_mean.unsqueeze(1)
            if x_std.dim() == 2: x_std = x_std.unsqueeze(1)
        x_norm = (x - x_mean) / (x_std + 1e-8)
        if mu.dim() == 2 and x_norm.dim() == 3: mu = mu.unsqueeze(1)
        if sigma.dim() == 2 and x_norm.dim() == 3: sigma = sigma.unsqueeze(1)
        sigma = torch.relu(sigma) + 1e-8
        return sigma * x_norm + mu

# --- MultiDomainPLEFENDModel 类定义 ---
class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,
                 bert_path_or_name,
                 clip_path_or_name,
                 out_channels, dropout, use_cuda=True,
                 text_token_len=197, image_token_len=197,
                 enable_hier_modules=False,
                 enable_cross_modal_calibration=False,
                 enable_hier_consistency=False,
                 enable_hier_fusion=False,
                 enable_calibration_loss=False,
                 calibration_temperature=0.5):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.use_cuda = use_cuda; self.num_expert = 6; self.domain_num = 2; self.num_share = 1
        self.unified_dim = 768; self.text_dim = 768; self.image_dim = 768
        self.text_token_len_expected = text_token_len; self.image_token_len_expected = image_token_len + 1 # MAE adds CLS token
        self.bert_path = bert_path_or_name; self.clip_path = clip_path_or_name

        # BERT
        try:
            logger.info(f"Loading BERT: {self.bert_path}")
            self.bert = BertModel.from_pretrained(self.bert_path, local_files_only=True)
            logger.info("BERT loaded.")
            for p in self.bert.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.bert = self.bert.cuda()
        except Exception as e:
            logger.error(f"Failed BERT load {self.bert_path}: {e}")
            self.bert = None
        # MAE
        self.model_size = "base"; mae_cp = f'./mae_pretrain_vit_{self.model_size}.pth'
        try:
            self.image_model = models_mae.__dict__[f"mae_vit_{self.model_size}_patch16"](norm_pix_loss=False)
            if os.path.exists(mae_cp):
                logger.info(f"Loading MAE weights: {mae_cp}")
                cp = torch.load(mae_cp, map_location='cpu')
                sd = cp['model'] if 'model' in cp else cp
                lr_msg = self.image_model.load_state_dict(sd, strict=False)
                logger.info(f"MAE load result: {lr_msg}")
            else:
                logger.warning(f"MAE checkpoint not found {mae_cp}. Random init.")
            for p in self.image_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.image_model = self.image_model.cuda()
        except Exception as e:
            logger.exception(f"Failed MAE load: {e}")
            self.image_model = None
        # CLIP
        try:
            logger.info(f"Loading CLIP: {self.clip_path}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_path, local_files_only=True)
            logger.info("CLIP loaded.")
            for p in self.clip_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.clip_model = self.clip_model.cuda()
        except Exception as e:
            logger.error(f"Failed CLIP load {self.clip_path}: {e}")
            self.clip_model = None

        # Other layers
        # !!!关键修改点!!!
        # 原来的 fk = {1: 64} 会导致 cnn_extractor (来自 layers.py) 输出 (Batch, 64)
        # 这与 text_experts_feature_sum 的 (Batch, 320) 初始化冲突
        # 将 fk 修改为使其输出320维特征
        fk = {1: 320} # 例如，使用一个kernel_size=1的卷积，输出320个特征
        # 或者，如果您希望使用多个卷积核，确保它们的feature_num总和为320:
        # fk = {1: 100, 2: 120, 3: 100} # 示例: 100+120+100 = 320

        expert_count = self.num_expert # 6
        shared_count = expert_count * 2 # 12

        # Experts (using cnn_extractor which should now output 320-dim features)
        self.text_experts = nn.ModuleList([nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(expert_count)]) for _ in range(self.domain_num)])
        self.image_experts = nn.ModuleList([nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(expert_count)]) for _ in range(self.domain_num)]) # 假设 image_experts 也用同样的 cnn_extractor
        self.fusion_experts = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(expert_count)]) for _ in range(self.domain_num)])
        self.text_share_expert = nn.ModuleList([nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(shared_count)]) for _ in range(self.num_share)])
        self.image_share_expert = nn.ModuleList([nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(shared_count)]) for _ in range(self.num_share)])
        self.fusion_share_expert = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(shared_count)]) for _ in range(self.num_share)])

        # Gates
        gate_out_dim = expert_count + shared_count # 6 + 12 = 18
        fusion0_out_dim = expert_count * 3 # 6 * 3 = 18

        self.image_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, gate_out_dim), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])
        self.text_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(), nn.Linear(self.unified_dim, gate_out_dim), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])
        self.fusion_gate_list0 = nn.ModuleList([nn.Sequential(nn.Linear(320, 160), nn.SiLU(), nn.Linear(160, fusion0_out_dim), nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])

        # 使用 layers.py 中的 MaskAttention 和 TokenAttention (如果导入成功)
        # 否则使用占位符版本
        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.cross_modal_calibration = CrossModalCalibration(
            local_dim=self.unified_dim,
            global_dim=512,
            calib_dim=320,
            dropout=0.1
        )
        self.enable_hier_modules = enable_hier_modules
        self.enable_cross_modal_calibration = enable_cross_modal_calibration
        self.enable_hier_consistency = enable_hier_consistency
        self.enable_hier_fusion = enable_hier_fusion
        self.enable_calibration_loss = enable_calibration_loss
        self.hier_consistency = HierarchicalConsistencyVerifier(
            local_dim=self.unified_dim,
            global_dim=512,
            alpha=0.5,
            temperature=0.1,
            tau_init=0.0
        )
        self.hier_fusion = HierarchicalFusionNoiseFilter(
            local_dim=self.unified_dim,
            global_dim=512,
            fusion_dim=160
        )


        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.max_classifier = MLP(320, mlp_dims, dropout)

        h_dims = mlp_dims if mlp_dims else [348] # Default hidden dims for MLP_fusion if not provided
        self.MLP_fusion = MLP_fusion(960, 320, h_dims, dropout)
        self.domain_fusion = MLP_fusion(320, 320, h_dims, dropout)
        self.MLP_fusion0 = MLP_fusion(768*2, 768, h_dims, dropout)
        # clip_fusion in layers.py is called 'clip_fuion', ensure consistency or use MLP_fusion
        # Assuming MLP_fusion is generic enough or layers.py's clip_fuion is used if imported
        self.clip_fusion = MLP_fusion(1024, 320, h_dims, dropout)


        self.att_mlp_text = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_img = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], dropout)

        self.mapping_IS_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(), nn.Linear(self.unified_dim // 2, 1))
        self.mapping_IS_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(), nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(), nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(), nn.Linear(self.unified_dim // 2, 1))
        self.adaIN = AdaIN()
        self.calibration_temperature = calibration_temperature

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        labels = kwargs.get('label', None)
        image_for_mae = kwargs['image']
        clip_pixel_values = kwargs['clip_image']
        clip_input_ids = kwargs['clip_text']
        clip_attention_mask = kwargs.get('clip_attention_mask', None)
        batch_size = inputs.shape[0]
        device = inputs.device

        text_feature_seq, image_feature_seq, clip_image_embed, clip_text_embed = None, None, None, None

        if self.bert:
            try:
                bert_outputs = self.bert(input_ids=inputs, attention_mask=masks)
                text_feature_seq = bert_outputs.last_hidden_state
            except Exception as e:
                logger.error(f"BERT error: {e}")
                text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.unified_dim, device=device)
        else:
            text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.unified_dim, device=device)

        if self.image_model:
            try:
                image_feature_seq = self.image_model.forward_ying(image_for_mae)
            except Exception as e:
                logger.error(f"MAE error: {e}")
                image_feature_seq = torch.zeros(batch_size, self.image_token_len_expected, self.unified_dim, device=device)
        else:
            image_feature_seq = torch.zeros(batch_size, self.image_token_len_expected, self.unified_dim, device=device)

        clip_output_dim = 512
        if self.clip_model:
            try:
                with torch.no_grad():
                    clip_img_out = self.clip_model.get_image_features(pixel_values=clip_pixel_values)
                    clip_image_embed = clip_img_out / (clip_img_out.norm(dim=-1, keepdim=True) + 1e-8)
                    clip_txt_out = self.clip_model.get_text_features(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
                    clip_text_embed = clip_txt_out / (clip_txt_out.norm(dim=-1, keepdim=True) + 1e-8)
            except Exception as e:
                logger.error(f"CLIP error: {e}")
        if clip_image_embed is None: clip_image_embed = torch.zeros(batch_size, clip_output_dim, device=device)
        if clip_text_embed is None: clip_text_embed = torch.zeros(batch_size, clip_output_dim, device=device)

        text_atn_feature = self.text_attention(text_feature_seq, masks)
        image_atn_feature, _ = self.image_attention(image_feature_seq)
        clip_fusion_feature_in = torch.cat((clip_image_embed, clip_text_embed), dim=-1).float()
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature_in)

        if self.enable_cross_modal_calibration:
            text_feature_seq, image_feature_seq = self.cross_modal_calibration(
                text_feature_seq,
                image_feature_seq,
                clip_text_embed,
                clip_image_embed,
                text_mask=masks
            )

        cal_loss = None
        cons_loss = None
        consistency_scores = None
        hier_fusion_feature = None
        if self.enable_calibration_loss:
            if masks is None:
                text_pooled = text_feature_seq.mean(dim=1)
            else:
                text_mask = masks.unsqueeze(-1)
                text_pooled = (text_feature_seq * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
            image_pooled = image_feature_seq.mean(dim=1)
            temp = self.calibration_temperature
            text_probs = F.softmax(text_pooled / temp, dim=-1)
            image_log_probs = F.log_softmax(image_pooled / temp, dim=-1)
            cal_loss = F.kl_div(image_log_probs, text_probs, reduction="batchmean")
        s_l = None
        s_g = None
        if self.enable_hier_consistency:
            s_l, s_g, cons_loss = self.hier_consistency(
                text_feature_seq,
                image_feature_seq,
                clip_text_embed,
                clip_image_embed,
                text_mask=masks,
                labels=labels
            )
            consistency_scores = torch.stack([s_l, s_g], dim=-1)
        if self.enable_hier_fusion:
            if s_l is None or s_g is None:
                s_l = torch.zeros(text_feature_seq.size(0), device=text_feature_seq.device)
                s_g = torch.zeros(text_feature_seq.size(0), device=text_feature_seq.device)
            hier_fusion_feature = self.hier_fusion(
                text_feature_seq,
                image_feature_seq,
                clip_text_embed,
                clip_image_embed,
                s_l,
                s_g,
                text_mask=masks
            )
        domain_idx = 0
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        # Text Experts Calculation
        output_shape_text = (batch_size, 320); text_experts_feature_sum = torch.zeros(output_shape_text, device=device); text_gate_share_expert_value_sum = torch.zeros(output_shape_text, device=device)
        for j in range(self.num_expert):
            tmp = self.text_experts[domain_idx][j](text_feature_seq) # Expected shape (B, 320) with corrected fk
            gate_val_for_expert = text_gate_out[:, j].unsqueeze(1)   # Expected shape (B, 1)
            # Now (B, 320) += (B, 320) * (B, 1) should work
            text_experts_feature_sum += (tmp * gate_val_for_expert)
        for j in range(self.num_expert * 2):
            tmp = self.text_share_expert[0][j](text_feature_seq)
            gate_val_for_shared_expert = text_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            text_experts_feature_sum += (tmp * gate_val_for_shared_expert)
            text_gate_share_expert_value_sum += (tmp * gate_val_for_shared_expert)
        att_text = F.softmax(self.att_mlp_text(text_experts_feature_sum), dim=-1)
        text_gate_expert_value = [att_text[:, i].unsqueeze(1) * text_experts_feature_sum for i in range(2)]

        # Image Experts Calculation
        output_shape_image = (batch_size, 320); image_experts_feature_sum = torch.zeros(output_shape_image, device=device); image_gate_share_expert_value_sum = torch.zeros(output_shape_image, device=device)
        for j in range(self.num_expert):
            tmp = self.image_experts[domain_idx][j](image_feature_seq)
            gate_val_for_expert = image_gate_out[:, j].unsqueeze(1)
            image_experts_feature_sum += (tmp * gate_val_for_expert)
        for j in range(self.num_expert * 2):
            tmp = self.image_share_expert[0][j](image_feature_seq)
            gate_val_for_shared_expert = image_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            image_experts_feature_sum += (tmp * gate_val_for_shared_expert)
            image_gate_share_expert_value_sum += (tmp * gate_val_for_shared_expert)
        att_img = F.softmax(self.att_mlp_img(image_experts_feature_sum), dim=-1)
        image_gate_expert_value = [att_img[:, i].unsqueeze(1) * image_experts_feature_sum for i in range(2)]

        # Fusion Modality Processing
        text_for_fusion = text_gate_share_expert_value_sum; image_for_fusion = image_gate_share_expert_value_sum
        fusion_share_feature_in = torch.cat((clip_fusion_feature, text_for_fusion, image_for_fusion), dim=-1); fusion_share_feature = self.MLP_fusion(fusion_share_feature_in)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature); fusion_gate_out0 = self.fusion_gate_list0[domain_idx](fusion_gate_input0)
        output_shape_fusion = (batch_size, 320); fusion_experts_feature_sum = torch.zeros(output_shape_fusion, device=device)
        for n in range(self.num_expert):
            tmp = self.fusion_experts[domain_idx][n](fusion_share_feature)
            gate_val = fusion_gate_out0[:, n].unsqueeze(1)
            fusion_experts_feature_sum += (tmp * gate_val)
        for n in range(self.num_expert * 2):
            tmp = self.fusion_share_expert[0][n](fusion_share_feature)
            gate_val = fusion_gate_out0[:, self.num_expert + n].unsqueeze(1)
            fusion_experts_feature_sum += (tmp * gate_val)
        att_mm = F.softmax(self.att_mlp_mm(fusion_experts_feature_sum), dim=-1)
        fusion_gate_expert_value0 = [att_mm[:, i].unsqueeze(1) * fusion_experts_feature_sum for i in range(2)]

        # Final Classification
        text_final_features = text_gate_expert_value[0]; image_final_features = image_gate_expert_value[0]; fusion_final_features = fusion_gate_expert_value0[0]
        if hier_fusion_feature is not None:
            fusion_final_features = fusion_final_features + hier_fusion_feature
        text_logits = self.text_classifier(text_final_features).squeeze(-1); image_logits = self.image_classifier(image_final_features).squeeze(-1); fusion_logits = self.fusion_classifier(fusion_final_features).squeeze(-1)
        all_modality = text_final_features + image_final_features + fusion_final_features
        if hier_fusion_feature is not None:
            all_modality = all_modality + hier_fusion_feature
        final_logits = self.max_classifier(all_modality).squeeze(-1)
        return final_logits, text_logits, image_logits, fusion_logits, cons_loss, cal_loss, consistency_scores

# --- Trainer Class ---
class Trainer():
    def __init__(self, emb_dim, mlp_dims,
                 bert_path_or_name, clip_path_or_name,
                 use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict, weight_decay,
                 save_param_dir, early_stop=10, epoches=100,
                 metric_key_for_early_stop='acc',
                 use_hier_modules=False,
                 use_hier_consistency=False,
                 use_hier_fusion=False,
                 use_calibration_loss=False,
                 use_cross_modal_calibration=False,
                 consistency_weight=0.0,
                 calibration_weight=0.0,
                 calibration_temperature=0.5,
                 loss_type='bce',
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 eval_threshold=0.5,
                 tune_threshold=False,
                 tune_threshold_metric='F1',
                 threshold_min=0.1,
                 threshold_max=0.9,
                 threshold_step=0.05): # <--- 修改这里的默认值为 'acc'
        self.lr = lr; self.weight_decay = weight_decay; self.train_loader = train_loader
        self.test_loader = test_loader; self.val_loader = val_loader; self.early_stop = early_stop
        self.epoches = epoches; self.category_dict = category_dict; self.use_cuda = use_cuda
        self.emb_dim = emb_dim; self.mlp_dims = mlp_dims; self.dropout = dropout
        # self.metric_key_for_early_stop 将会是传入的值，或者默认的 'acc'
        self.metric_key_for_early_stop = metric_key_for_early_stop 
        self.save_param_dir = save_param_dir
        self.calibration_weight = calibration_weight
        self.consistency_weight = consistency_weight
        self.use_hier_modules = use_hier_modules
        self.use_hier_consistency = use_hier_consistency or use_hier_modules
        self.use_hier_fusion = use_hier_fusion or use_hier_modules
        self.use_calibration_loss = use_calibration_loss or use_hier_modules
        self.use_cross_modal_calibration = use_cross_modal_calibration
        self.calibration_temperature = calibration_temperature
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.eval_threshold = eval_threshold
        self.tune_threshold = tune_threshold
        self.tune_threshold_metric = tune_threshold_metric
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        os.makedirs(self.save_param_dir, exist_ok=True)

        self.model = MultiDomainPLEFENDModel(
            emb_dim=self.emb_dim, mlp_dims=self.mlp_dims,
            bert_path_or_name=bert_path_or_name,
            clip_path_or_name=clip_path_or_name,
            out_channels=320, # 这个参数在您当前的模型中似乎没有被直接使用
            dropout=self.dropout, use_cuda=self.use_cuda,
            enable_hier_modules=self.use_hier_modules,
            enable_cross_modal_calibration=self.use_cross_modal_calibration,
            enable_hier_consistency=self.use_hier_consistency,
            enable_hier_fusion=self.use_hier_fusion,
            enable_calibration_loss=self.use_calibration_loss,
            calibration_temperature=self.calibration_temperature
        )
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            logger.warning("CUDA not available/requested. Model on CPU.")

    def _focal_loss(self, logits, labels):
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = labels * probs + (1.0 - labels) * (1.0 - probs)
        alpha_t = labels * self.focal_alpha + (1.0 - labels) * (1.0 - self.focal_alpha)
        loss = alpha_t * ((1.0 - p_t) ** self.focal_gamma) * bce
        return loss.mean()

    def _get_threshold_grid(self):
        if self.threshold_step <= 0:
            return [self.eval_threshold]
        grid = np.arange(self.threshold_min, self.threshold_max + 1e-12, self.threshold_step)
        if grid.size == 0:
            return [self.eval_threshold]
        return [float(x) for x in grid]

    def _select_threshold(self, label_list, pred_probs, category_list):
        best_threshold = self.eval_threshold
        best_metrics = None
        metric_key = self.tune_threshold_metric or 'F1'
        best_metric = -1.0
        for threshold in self._get_threshold_grid():
            metrics = calculate_metrics(
                label_list,
                pred_probs,
                category_list,
                self.category_dict,
                threshold=threshold
            )
            metric_val = metrics.get(metric_key, 0.0)
            if metric_val > best_metric:
                best_metric = metric_val
                best_threshold = threshold
                best_metrics = metrics
        return best_threshold, best_metrics or {}

    def train(self):
        use_focal = self.loss_type == 'focal'
        if self.loss_type not in ('bce', 'focal'):
            logger.warning(f"Unknown loss_type '{self.loss_type}', fallback to bce.")
            use_focal = False
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        # Recorder 将使用 self.metric_key_for_early_stop (现在默认为 'acc' 或传入的值)
        recorder = Recorder(self.early_stop, metric_key=self.metric_key_for_early_stop) 

        for epoch in range(self.epoches):
            self.model.train(); train_data_iter = tqdm.tqdm(self.train_loader); avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                try:
                    batch_data = clipdata2gpu(batch)
                    if batch_data is None:
                        logger.warning(f"Skipping batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    label = batch_data.get('label')
                    if label is None:
                        logger.warning(f"Skipping batch {step_n} due to missing label.")
                        continue

                    final_logits, text_logits, image_logits, fusion_logits, cons_loss, cal_loss, _ = self.model(**batch_data)

                    if use_focal:
                        loss0 = self._focal_loss(final_logits, label)
                        loss1 = self._focal_loss(text_logits, label)
                        loss2 = self._focal_loss(image_logits, label)
                        loss3 = self._focal_loss(fusion_logits, label)
                    else:
                        loss0 = loss_fn(final_logits, label.float())
                        loss1 = loss_fn(text_logits, label.float())
                        loss2 = loss_fn(image_logits, label.float())
                        loss3 = loss_fn(fusion_logits, label.float())
                    loss = loss0 + (loss1 + loss2 + loss3) / 3.0
                    if self.consistency_weight > 0 and cons_loss is not None:
                        loss = loss + self.consistency_weight * cons_loss
                    if self.calibration_weight > 0 and cal_loss is not None:
                        loss = loss + self.calibration_weight * cal_loss
                    optimizer.zero_grad()
                    if not torch.isfinite(loss):
                        logger.warning(f"Non-finite loss at epoch {epoch+1} step {step_n}: {loss.item()}. Skipping update.")
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    avg_loss.add(loss.item())
                    train_data_iter.set_description(f"Epoch {epoch+1}/{self.epoches}")
                    train_data_iter.set_postfix(loss=avg_loss.item(), lr=optimizer.param_groups[0]['lr'])
                except Exception as e:
                    if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                         logger.error(f"Tensor mismatch error at Train step {epoch}-{step_n}: {e}", exc_info=True)
                    # 更详细地记录图像处理相关的警告/错误
                    elif "collate_fn" in str(e) or "image" in str(e).lower() or "channel" in str(e).lower():
                         logger.error(f"Image processing related error at Train step {epoch}-{step_n}: {e}", exc_info=True)
                    else:
                        logger.exception(f"Train step {step_n} error: {e}")
                    continue
            
            if scheduler is not None: scheduler.step()
            logger.info(f'Train Epoch {epoch+1} Done; Avg Loss: {avg_loss.item():.4f}; LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if self.val_loader is None:
                logger.warning("Val loader not provided, skipping validation.")
                # 如果没有验证集，可以考虑在此处保存最后一个epoch的模型，或者不进行任何基于验证的保存
                # last_epoch_model_path = os.path.join(self.save_param_dir, f'model_epoch_{epoch+1}.pth')
                # torch.save(self.model.state_dict(), last_epoch_model_path)
                # logger.info(f"Saved model from last training epoch to {last_epoch_model_path}")
                continue # 跳过验证步骤

            # Validation logic
            try:
                val_results = self.test(self.val_loader, tune_threshold=self.tune_threshold)
                if not val_results: # 检查 val_results 是否为空或无效
                    logger.warning(f"Val epoch {epoch+1} did not return valid results. Skipping score processing for this epoch.")
                    continue

                # 从 val_results 中安全地获取指标，如果键不存在则默认为0.0
                current_metric_val = val_results.get(self.metric_key_for_early_stop, 0.0)
                acc_val = val_results.get('acc', 0.0)
                f1_val = val_results.get('F1', 0.0)
                auc_val = val_results.get('auc', 0.0)
                
                logger.info(
                    f"Val E{epoch+1}: Acc:{acc_val:.4f} F1:{f1_val:.4f} AUC:{auc_val:.4f} "
                    f"Tracked({self.metric_key_for_early_stop}):{current_metric_val:.4f} "
                    f"Thr:{self.eval_threshold:.2f}"
                )
                real_metrics = val_results.get('Real', val_results.get('real', {}))
                fake_metrics = val_results.get('Fake', val_results.get('fake', {}))
                logger.info(f"  Real:{repr(real_metrics)}, Fake:{repr(fake_metrics)}")
                
                mark = recorder.add(val_results) # recorder.add 现在应该能处理空的val_results了（如果它被设计成这样）
                                                 # 或者确保 val_results 总是有 Recorder期望的键
                if mark == 'save':
                    save_p = os.path.join(self.save_param_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_p)
                    logger.info(f"Best model saved based on '{self.metric_key_for_early_stop}': {save_p}")
                elif mark == 'esc':
                    logger.info(f"Early stopping triggered based on '{self.metric_key_for_early_stop}'.")
                    break 
            except Exception as e:
                logger.exception(f"Val epoch {epoch+1} error: {e}")
                continue

        logger.info("Training loop finished."); recorder.showfinal()
        best_model_path = os.path.join(self.save_param_dir, 'best_model.pth'); loaded_best = False
        final_model_to_test_path = best_model_path
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model for final test: {best_model_path}")
            try:
                # 确保在加载状态字典之前，模型实例已经存在并且结构匹配
                self.model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
                loaded_best = True
            except Exception as e:
                logger.error(f"Failed to load best model state_dict: {e}. Using model state from end of training.")
        
        if not loaded_best: # 如果最佳模型没有成功加载
            final_model_to_test_path = os.path.join(self.save_param_dir, 'final_training_state.pth')
            logger.warning(f"Best model not loaded. Testing with model state from end of training. Saving this state to: {final_model_to_test_path}")
            try:
                torch.save(self.model.state_dict(), final_model_to_test_path)
            except Exception as es:
                logger.error(f"Failed to save final training state model: {es}")


        final_results = None
        if self.test_loader is None:
            logger.warning("Test loader not provided. Skipping final test.")
            final_results = recorder.max if hasattr(recorder, 'max') else None
        else:
            logger.info("Starting final test with the chosen model...");
            try:
                final_results = self.test(self.test_loader, threshold=self.eval_threshold) # 调用 test, 返回包含详细指标的字典
                if final_results:
                    # 打印总体指标
                    acc = final_results.get('acc',0.0); f1=final_results.get('F1',0.0); auc=final_results.get('auc',0.0)
                    precision = final_results.get('precision', 0.0); recall = final_results.get('recall', 0.0)
                    logger.info(f"Final Test Results: Acc:{acc:.4f} F1:{f1:.4f} AUC:{auc:.4f} Precision:{precision:.4f} Recall:{recall:.4f}")

                    # --- !!! 关键修改：打印 Real/Fake 指标 !!! ---
                    real_m = final_results.get('Real', {}) # 从结果中获取 'Real' 子字典
                    fake_m = final_results.get('Fake', {}) # 从结果中获取 'Fake' 子字典

                    # 使用 .get('metric', 0.0) 来安全地获取值，如果键不存在则显示 0.0
                    log_final_class_summary = (
                        f"  Real (label=0): P:{real_m.get('precision', 0.0):.4f} "
                        f"R:{real_m.get('recall', 0.0):.4f} "
                        f"F1:{real_m.get('F1', 0.0):.4f} | "
                        f"Fake (label=1): P:{fake_m.get('precision', 0.0):.4f} "
                        f"R:{fake_m.get('recall', 0.0):.4f} "
                        f"F1:{fake_m.get('F1', 0.0):.4f}"
                    )
                    logger.info(log_final_class_summary) # 打印详细的分类指标

                else:
                    logger.error("Final test did not return valid results.")
            except Exception as e:
                logger.exception(f"Final test execution error: {e}")
        return final_results, final_model_to_test_path


    def test(self, dataloader, threshold=None, tune_threshold=False):
        pred_probs, label_list, category_list = [], [], []
        if dataloader is None:
            logger.error("Test dataloader is None."); return {}
        self.model.eval(); data_iter = tqdm.tqdm(dataloader, desc="Testing")
        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                try:
                    batch_data = clipdata2gpu(batch)
                    if batch_data is None: 
                        logger.warning(f"Skipping test batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    batch_label = batch_data.get('label')
                    batch_category = batch_data.get('category') # 在calculate_metrics中会用到
                    if batch_label is None: # category 不是绝对必要，但label是
                        logger.warning(f"Skipping test batch {step_n} due to missing label.")
                        continue
                    if batch_category is None: # 如果category_dict非空，则category也最好有
                         logger.debug(f"Test batch {step_n} missing category, will pass None to metrics if category_dict used.")


                    final_logits, _, _, _, _, _, _ = self.model(**batch_data)
                    batch_pred_prob = torch.sigmoid(final_logits)
                    label_list.extend(batch_label.cpu().numpy().tolist())
                    pred_probs.extend(batch_pred_prob.cpu().numpy().tolist())
                    if batch_category is not None:
                        category_list.extend(batch_category.cpu().numpy().tolist())
                    else: # 如果category缺失，但calculate_metrics需要它，则填充一个占位符
                        category_list.extend([None] * batch_label.size(0))


                except Exception as e:
                    logger.exception(f"Test batch {step_n} error: {e}")
                    continue # 继续下一个batch
        
        if not label_list or not pred_probs: # 确保有数据用于计算指标
            logger.warning("No data was successfully processed in test function to calculate metrics.")
            return {}
        
        # 确保 category_list 与 label_list 长度一致，即使有些是 None
        if len(category_list) != len(label_list) and self.category_dict:
             logger.warning(f"Mismatch in length of category_list ({len(category_list)}) and label_list ({len(label_list)}). Filling with None.")
             category_list = (category_list + [None] * len(label_list))[:len(label_list)]


        try:
            if tune_threshold:
                best_threshold, metric_res = self._select_threshold(label_list, pred_probs, category_list)
                self.eval_threshold = best_threshold
                metric_key = self.tune_threshold_metric or 'F1'
                metric_val = metric_res.get(metric_key, 0.0)
                logger.info(f"Threshold tuned. Best threshold: {best_threshold:.2f} (metric={metric_key}:{metric_val:.4f})")
            else:
                use_threshold = self.eval_threshold if threshold is None else threshold
                if self.category_dict:
                    metric_res = calculate_metrics(label_list, pred_probs, category_list, self.category_dict, threshold=use_threshold)
                else:
                    metric_res = calculate_metrics(label_list, pred_probs, threshold=use_threshold)
        except Exception as e:
            logger.exception(f"Metrics calculation error: {e}"); metric_res = {}
        return metric_res
