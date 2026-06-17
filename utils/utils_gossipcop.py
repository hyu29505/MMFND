
# 请将这段代码放在您的 utils/utils_gossipcop.py 文件中

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import numpy as np

logger = logging.getLogger(__name__) # 确保 logger 已定义

def clipdata2gpu(batch_input):
    """
    将批次数据中的张量移动到 GPU。
    'batch_input' 可以是由 DataLoader 返回的字典或元组/列表。
    """
    if batch_input is None:
        logger.warning("clipdata2gpu received None batch_input.")
        return None

    batch_dict = None

    if isinstance(batch_input, dict):
        batch_dict = batch_input
    elif isinstance(batch_input, (list, tuple)):
        # 兼容来自不同dataloader的元组/列表格式
        if len(batch_input) == 8: # 来自 bert_data TensorDataset 的项目数量
            keys = [
                'content', 'content_masks', 'label', 'category',
                'image', 'clip_image', 'clip_text', 'clip_attention_mask'
            ]
            batch_dict = dict(zip(keys, batch_input))
        else:
            logger.error(f"clipdata2gpu received a tuple/list with unexpected number of items: {len(batch_input)}.")
            return None
    else:
        logger.error(f"clipdata2gpu expects batch_input to be a dictionary, tuple, or list, but received {type(batch_input)}.")
        return None

    if batch_dict is None:
        logger.error("clipdata2gpu: batch_dict is None after type checking.")
        return None

    gpu_batch = {}
    try:
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.cuda()
            else:
                gpu_batch[key] = value # 非张量数据保持原样
        return gpu_batch
    except Exception as e:
        logger.exception(f"clipdata2gpu encountered an unexpected error: {e}")
        return None

# --- 您可能还有其他的工具函数，保持它们不变 ---
class Averager:
    def __init__(self): self.n=0.0; self.v=0.0
    def add(self, x): self.v=(self.v*self.n+x)/(self.n+1); self.n+=1
    def item(self): return self.v

class Recorder:
    def __init__(self, early_stop_patience=10, metric_key='F1'):
        self.max = {metric_key: 0.0} # 使用传入的 key 初始化
        self.cur = {metric_key: 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop_patience = early_stop_patience
        self.metric_key = metric_key # 保存用于比较的 key

    def add(self, res):
        if self.metric_key not in res:
            logger.warning(f"Recorder: 结果字典中缺少关键指标 '{self.metric_key}'。无法进行比较。")
            return 'continue'

        self.cur = res
        self.curindex += 1
        
        if self.cur[self.metric_key] > self.max.get(self.metric_key, -1): # 使用.get以避免初始max字典没有metric_key的情况
            self.max = self.cur
            self.maxindex = self.curindex

            # --- !!! 关键修改点 START !!! ---
            # 构建一个包含多个核心指标的详细日志字符串
            metrics_str = (
                f"Acc: {self.max.get('acc', 0.0):.4f}, "
                f"AUC: {self.max.get('auc', 0.0):.4f}, "
                f"Precision: {self.max.get('precision', 0.0):.4f}, "
                f"Recall: {self.max.get('recall', 0.0):.4f}, "
                f"F1: {self.max.get('F1', 0.0):.4f}"
            )
            # 使用新的字符串格式化日志信息
            logger.info(f"Recorder: 新的最佳结果! Epoch {self.curindex} (Tracked: {self.metric_key}={self.max[self.metric_key]:.4f}). Details: {metrics_str}")
            # --- !!! 关键修改点 END !!! ---
            
            return 'save'
        elif self.curindex - self.maxindex >= self.early_stop_patience:
            logger.info(f"Recorder: 触发早停。连续 {self.early_stop_patience} 个 epoch 没有提升 (基于 '{self.metric_key}')。")
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        logger.info("--- Recorder 最终结果 ---")
        logger.info(f"最佳指标 ({self.metric_key}) 在第 {self.maxindex} 个 epoch 达到:")
        if self.max: # 确保 self.max 不是空的
            for key, val in self.max.items():
                if isinstance(val, float): logger.info(f"  {key}: {val:.4f}")
                elif isinstance(val, dict): logger.info(f"  {key}: {val}")
                else: logger.info(f"  {key}: {val}")
        else:
            logger.warning("  没有记录到有效的最佳结果。")


def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None, threshold=0.5):
    """
    计算各种评估指标，包括总体指标和按类别（Real/Fake）细分的指标。
    与 Weibo 统一语义：1=Fake, 0=Real（此处会对原始标签/概率做反转）。
    threshold: 用于将概率转为类别的阈值，默认 0.5。
    """
    if not isinstance(label_list, np.ndarray): label_list = np.array(label_list)
    if not isinstance(pred_probs, np.ndarray): pred_probs = np.array(pred_probs)

    if not label_list.size or not pred_probs.size or len(label_list) != len(pred_probs):
        logger.warning("calculate_metrics: 标签列表或预测概率列表为空或长度不匹配。")
        return {}

    # --- 统一为 Weibo 语义：1=Fake, 0=Real ---
    # GossipCop 原始标签/预测是 1=Real, 0=Fake，这里做反转
    label_list = 1 - label_list
    pred_probs = 1 - pred_probs

    # --- 计算总体指标 ---
    pred_labels = (pred_probs >= threshold).astype(int)
    metrics = {}
    metrics['acc'] = accuracy_score(label_list, pred_labels)
    # 总体 Precision, Recall, F1 计算正类 (Fake=1) 的指标
    metrics['precision'] = precision_score(label_list, pred_labels, pos_label=1, zero_division=0)
    metrics['recall'] = recall_score(label_list, pred_labels, pos_label=1, zero_division=0)
    metrics['F1'] = f1_score(label_list, pred_labels, pos_label=1, zero_division=0)
    try:
        # AUC 需要数据中同时包含 0 和 1 两类标签
        if len(np.unique(label_list)) > 1:
            metrics['auc'] = roc_auc_score(label_list, pred_probs)
        else:
            logger.warning(f"calculate_metrics: 数据中只存在单一类别标签，无法计算 AUC。")
            metrics['auc'] = 0.0
    except ValueError as e:
        logger.warning(f"计算 AUC 时出错: {e}")
        metrics['auc'] = 0.0

    # --- 计算 Real 和 Fake 类别的指标（Weibo 语义） ---
    # Fake 类 (标签=1)
    if np.any(label_list == 1):
        metrics['Fake'] = {
            'precision': precision_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'recall': recall_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'F1': f1_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'support': int(np.sum(label_list == 1))
        }
    else:
        metrics['Fake'] = {'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'support': 0}

    # Real 类 (标签=0)
    if np.any(label_list == 0):
        metrics['Real'] = {
            'precision': precision_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'recall': recall_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'F1': f1_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'support': int(np.sum(label_list == 0))
        }
    else:
        metrics['Real'] = {'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'support': 0}

    # --- 统一键名：提供 lowercase 版本，便于与 Weibo 日志对齐 ---
    # 额外提供 real/fake，避免下游兼容问题
    metrics['real'] = metrics.get('Real', {})
    metrics['fake'] = metrics.get('Fake', {})

    # --- (可选) 处理 category_dict 的逻辑（如果需要） ---
    # 这部分逻辑与 Real/Fake 指标计算是独立的
    if category_list is not None and category_dict is not None and len(category_list) == len(label_list):
        category_list = np.array(category_list)
        for category_name, category_id in category_dict.items():
             mask = (category_list == category_id)
             cat_labels = label_list[mask]
             cat_pred_labels = pred_labels[mask]
             cat_pred_probs = np.array(pred_probs)[mask]

             if len(cat_labels) > 0:
                  cat_metrics = {}
                  cat_metrics['acc'] = accuracy_score(cat_labels, cat_pred_labels)
                  cat_metrics['precision'] = precision_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0) # 正类=Fake(1)
                  cat_metrics['recall'] = recall_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0)
                  cat_metrics['F1'] = f1_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0)
                  try:
                      if len(np.unique(cat_labels)) > 1: cat_metrics['auc'] = roc_auc_score(cat_labels, cat_pred_probs)
                      else: cat_metrics['auc'] = 0.0
                  except ValueError: cat_metrics['auc'] = 0.0
                  metrics[category_name] = cat_metrics # 例如 metrics['gossip'] = {...}

    return metrics