# # # -*-codeing = utf-8 -*-

# # # -*-codeing = utf-8 -*-

# import torch
# from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
# import numpy as np

# # --- START: MODIFIED CODE ---
# # 修改了 clipdata2gpu 和 data2gpu 函数，使其能够接收 use_cuda 参数

# def clipdata2gpu(batch, use_cuda):
#     """
#     根据 use_cuda 标志，有条件地将数据移动到 GPU。
#     """
#     batch_data = {
#         'content': batch[0],
#         'content_masks': batch[1],
#         'label': batch[2],
#         'category': batch[3],
#         'image': batch[4],
#         'clip_image': batch[5],
#         'clip_text': batch[6]
#     }
#     if use_cuda:
#         for key, value in batch_data.items():
#             if isinstance(value, torch.Tensor):
#                 batch_data[key] = value.cuda()
    
#     # 关键修改：为蒸馏任务添加 teacher reasoning embeddings 的处理
#     # 假设它们在 batch 的第7、8、9个位置
#     if len(batch) > 7 and batch[7] is not None:
#         batch_data['teacher_reasoning_text_emb'] = batch[7].cuda() if use_cuda else batch[7]
#     if len(batch) > 8 and batch[8] is not None:
#         batch_data['teacher_reasoning_image_emb'] = batch[8].cuda() if use_cuda else batch[8]
#     if len(batch) > 9 and batch[9] is not None:
#         batch_data['teacher_reasoning_cross_emb'] = batch[9].cuda() if use_cuda else batch[9]

#     return batch_data

# def data2gpu(batch, use_cuda):
#     """
#     根据 use_cuda 标志，有条件地将数据移动到 GPU。
#     """
#     batch_data = {
#         'content': batch[0],
#         'content_masks': batch[1],
#         'label': batch[2],
#         'category': batch[3],
#         'image': batch[4]
#     }
#     if use_cuda:
#         for key, value in batch_data.items():
#             if isinstance(value, torch.Tensor):
#                 batch_data[key] = value.cuda()
#     return batch_data

# # --- END: MODIFIED CODE ---


# class Averager():

#     def __init__(self):
#         self.n = 0
#         self.v = 0

#     def add(self, x):
#         self.v = (self.v * self.n + x) / (self.n + 1)
#         self.n += 1

#     def item(self):
#         return self.v


# def metrics(y_true, y_pred, category, category_dict):
#     res_by_category = {}
#     metrics_by_category = {}
#     reverse_category_dict = {}
#     for k, v in category_dict.items():
#         reverse_category_dict[v] = k
#         res_by_category[k] = {"y_true": [], "y_pred": []}

#     # 处理 category 可能为空列表的情况
#     if category:
#         for i, c in enumerate(category):
#             c_val = c.item() if hasattr(c, 'item') else c
#             category_name = reverse_category_dict[c_val]
#             res_by_category[category_name]['y_true'].append(y_true[i])
#             res_by_category[category_name]['y_pred'].append(y_pred[i])

#     # 计算整体指标
#     try:
#         metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
#     except ValueError:
#         metrics_by_category['auc'] = 0.0

#     y_pred_int = np.around(np.array(y_pred)).astype(int)
#     metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
#     metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
#     metrics_by_category['f1'] = metrics_by_category['metric'] # 别名 F1

#     # 计算各类别指标
#     for c, res in res_by_category.items():
#         if not res['y_true']:
#             continue
        
#         y_pred_cat_int = np.around(np.array(res['y_pred'])).astype(int)
        
#         cat_auc = 0.0
#         try:
#             # AUC 需要至少两个类别
#             if len(np.unique(res['y_true'])) > 1:
#                 cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
#         except ValueError:
#             pass

#         metrics_by_category[c] = {
#             'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'auc': cat_auc,
#             'acc': accuracy_score(res['y_true'], y_pred_cat_int)
#         }
#     return metrics_by_category


# def metricsTrueFalse(y_true, y_pred, category, category_dict):
#     """
#     此函数现在直接调用修正后的 metrics 函数，以提供统一和健壮的格式。
#     """
#     results = metrics(y_true, y_pred, category, category_dict)

#     # 分别计算真新闻和假新闻的指标
#     y_true_np = np.array(y_true)
#     y_pred_int = np.around(np.array(y_pred)).astype(int)

#     # 真实新闻 (label=0)
#     real_indices = np.where(y_true_np == 0)[0]
#     if len(real_indices) > 0:
#         real_true = y_true_np[real_indices]
#         real_pred = y_pred_int[real_indices]
#         results['Real_Acc'] = accuracy_score(real_true, real_pred)
#         results['Real_Pre'] = precision_score(real_true, real_pred, pos_label=0, zero_division=0)
#         results['Real_Rec'] = recall_score(real_true, real_pred, pos_label=0, zero_division=0)
#         results['Real_F1'] = f1_score(real_true, real_pred, pos_label=0, zero_division=0)

#     # 虚假新闻 (label=1)
#     fake_indices = np.where(y_true_np == 1)[0]
#     if len(fake_indices) > 0:
#         fake_true = y_true_np[fake_indices]
#         fake_pred = y_pred_int[fake_indices]
#         results['Fake_Acc'] = accuracy_score(fake_true, fake_pred)
#         results['Fake_Pre'] = precision_score(fake_true, fake_pred, pos_label=1, zero_division=0)
#         results['Fake_Rec'] = recall_score(fake_true, fake_pred, pos_label=1, zero_division=0)
#         results['Fake_F1'] = f1_score(fake_true, fake_pred, pos_label=1, zero_division=0)

#     # 宏平均指标
#     results['Macro_Acc'] = results.get('acc', 0)
#     results['Macro_Pre'] = results.get('precision', 0)
#     results['Macro_Rec'] = results.get('recall', 0)
#     results['Macro_F1'] = results.get('f1', 0)
    
#     return results


# class Recorder():

#     def __init__(self, early_step, metric_key='metric'):
#         self.max = {metric_key: 0}
#         self.cur = {metric_key: 0}
#         self.maxindex = 0
#         self.curindex = 0
#         self.early_step = early_step
#         self.metric_key = metric_key

#     def add(self, x):
#         self.cur = x
#         self.curindex += 1
#         print("current", self.cur)
#         return self.judge()

#     def judge(self):
#         # 确保比较的键存在
#         if self.cur.get(self.metric_key, 0) > self.max.get(self.metric_key, 0):
#             self.max = self.cur
#             self.maxindex = self.curindex
#             self.showfinal()
#             return 'save'
#         self.showfinal()
#         if self.curindex - self.maxindex >= self.early_step:
#             return 'esc'
#         else:
#             return 'continue'

#     def showfinal(self):
#         print("Max", self.max)

# # # -*-codeing = utf-8 -*-

# # import torch
# # from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
# # import numpy as np

# # ## Data Handling
# # def data_to_gpu(data, use_cuda):
# #     """
# #     Moves a dictionary or list containing tensors to the GPU, if available.
# #     """
# #     if not use_cuda:
# #         return data

# #     if isinstance(data, dict):
# #         for key, value in data.items():
# #             if isinstance(value, torch.Tensor):
# #                 data[key] = value.cuda()
# #     elif isinstance(data, list):
# #         data = [item.cuda() if isinstance(item, torch.Tensor) else item for item in data]
    
# #     return data

# # ## Metrics Calculation
# # def metrics(y_true, y_pred, category, category_dict):
# #     """
# #     This helper function calculates various metrics and organizes them by category.
# #     """
# #     res_by_category = {}
# #     metrics_by_category = {}
# #     reverse_category_dict = {}
# #     for k, v in category_dict.items():
# #         reverse_category_dict[v] = k
# #         res_by_category[k] = {"y_true": [], "y_pred": []}

# #     # Populates the res_by_category dictionary with true and predicted values for each category
# #     if category:
# #         for i, c in enumerate(category):
# #             c_val = c.item() if hasattr(c, 'item') else c
# #             category_name = reverse_category_dict[c_val]
# #             res_by_category[category_name]['y_true'].append(y_true[i])
# #             res_by_category[category_name]['y_pred'].append(y_pred[i])

# #     # Calculates overall metrics
# #     try:
# #         metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
# #     except ValueError:
# #         metrics_by_category['auc'] = 0.0
# #         pass
    
# #     y_pred_int = np.around(np.array(y_pred)).astype(int)
# #     metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
# #     metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
# #     metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
# #     metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
# #     metrics_by_category['f1'] = metrics_by_category['metric']

# #     # Calculates metrics by category
# #     for c, res in res_by_category.items():
# #         if not res['y_true']:
# #             continue
        
# #         y_pred_cat_int = np.around(np.array(res['y_pred'])).astype(int)
        
# #         cat_auc = 0.0
# #         try:
# #             if len(np.unique(res['y_true'])) > 1:
# #                 cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
# #         except ValueError:
# #             pass

# #         metrics_by_category[c] = {
# #             'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
# #             'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
# #             'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
# #             'auc': cat_auc,
# #             'acc': accuracy_score(res['y_true'], y_pred_cat_int)
# #         }

# #     return metrics_by_category


# # def metricsTrueFalse(y_true, y_pred, category, category_dict):
# #     """
# #     This function calls the metrics function to get the output format.
# #     """
# #     return metrics(y_true, y_pred, category, category_dict)

# # ## Training Utilities
# # class Averager():
# #     """
# #     A class to calculate the running average of a value.
# #     """
# #     def __init__(self):
# #         self.n = 0
# #         self.v = 0

# #     def add(self, x):
# #         self.v = (self.v * self.n + x) / (self.n + 1)
# #         self.n += 1

# #     def item(self):
# #         return self.v


# # class Recorder():
# #     """
# #     A class to record and manage model training progress for early stopping.
# #     """
# #     def __init__(self, early_step):
# #         self.max = {'metric': 0}
# #         self.cur = {'metric': 0}
# #         self.maxindex = 0
# #         self.curindex = 0
# #         self.early_step = early_step

# #     def add(self, x):
# #         self.cur = x
# #         self.curindex += 1
# #         print("curent", self.cur)
# #         return self.judge()

# #     def judge(self):
# #         if self.cur['metric'] > self.max['metric']:
# #             self.max = self.cur
# #             self.maxindex = self.curindex
# #             self.showfinal()
# #             return 'save'
# #         self.showfinal()
# #         if self.curindex - self.maxindex >= self.early_step:
# #             return 'esc'
# #         else:
# #             return 'continue'

# #     def showfinal(self):
# #         print("Max", self.max)


# 请将这段代码放在您的 utils/utils.py 文件中

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
        logger.debug("clipdata2gpu received a dictionary batch.")
    elif isinstance(batch_input, (list, tuple)):
        logger.debug(f"clipdata2gpu received a tuple/list batch with {len(batch_input)} items.")
        # 假设这个结构来自 bert_data 的 TensorDataset
        # 顺序: bert_ids, bert_masks, label, category, mae_image, clip_image_pixels, clip_text_ids, clip_text_masks
        if len(batch_input) == 8: # 来自 bert_data TensorDataset 的项目数量
            keys = [
                'content', 'content_masks', 'label', 'category',
                'image', 'clip_image', 'clip_text', 'clip_attention_mask'
            ]
            batch_dict = dict(zip(keys, batch_input))
            logger.debug("clipdata2gpu converted tuple/list batch to dictionary.")
        else:
            logger.error(f"clipdata2gpu received a tuple/list with unexpected number of items: {len(batch_input)}. Expected 8 for bert_data tuple or a dictionary.")
            # 如果你的 FakeNet_dataset 也可能输出元组但长度不同，你需要调整此处的逻辑或让 FakeNet_dataset 固定输出字典
            return None # 或者根据你的具体情况决定如何处理
    else:
        logger.error(f"clipdata2gpu expects batch_input to be a dictionary, tuple, or list, but received {type(batch_input)}.")
        return None

    if batch_dict is None:
        logger.error("clipdata2gpu: batch_dict is None after type checking. This should not happen if logic is correct.")
        return None

    gpu_batch = {}
    try:
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.cuda()
                # logger.debug(f"Moved tensor for key '{key}' to GPU.")
            else:
                gpu_batch[key] = value # 非张量数据保持原样
                logger.debug(f"Key '{key}' in batch is not a Tensor (type: {type(value)}), not moved to GPU.")
        return gpu_batch
    except AttributeError as e:
        logger.error(f"clipdata2gpu error moving data to GPU (AttributeError, possibly None or non-Tensor for a key): {e}")
        logger.error(f"Offending batch_dict (content types):")
        for k, v_type in ((k_in, type(v_in)) for k_in, v_in in batch_dict.items()):
            logger.error(f"  Key '{k_type}': Type {v_type}")
        return None
    except Exception as e:
        # 使用 logger.exception 可以记录完整的堆栈跟踪信息
        logger.exception(f"clipdata2gpu encountered an unexpected error: {e}")
        return None

# --- 您可能还有其他的工具函数，保持它们不变 ---
class Averager:
     # ... (Averager 类的实现) ...
     def __init__(self): self.n=0.0; self.v=0.0
     def add(self, x): self.v=(self.v*self.n+x)/(self.n+1); self.n+=1
     def item(self): return self.v

class Recorder:
    # ... (Recorder 类的实现) ...
    def __init__(self, early_stop_patience=10, metric_key='F1'):
        self.max = {metric_key: 0.0} # 使用传入的 key 初始化
        self.cur = {metric_key: 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop_patience = early_stop_patience
        self.metric_key = metric_key # 保存用于比较的 key

    def add(self, res):
        # 确保 res 字典中包含我们关心的 metric_key
        if self.metric_key not in res:
            logger.warning(f"Recorder: 结果字典中缺少关键指标 '{self.metric_key}'。无法进行比较。")
            return 'continue' # 或者其他表示无法比较的状态

        self.cur = res
        self.curindex += 1
        
        # 使用 self.metric_key 进行比较
        if self.cur[self.metric_key] > self.max[self.metric_key]:
            self.max = self.cur
            self.maxindex = self.curindex
            logger.info(f"Recorder: 新的最佳结果 ({self.metric_key}={self.max[self.metric_key]:.4f}) 在 epoch {self.curindex}")
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
                  else: logger.info(f"  {key}: {val}") # 对于非浮点值直接打印
        else:
             logger.warning("  没有记录到有效的最佳结果。")


def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None):
    """
    计算各种评估指标，包括总体指标和按类别（Real/Fake）细分的指标。
    假定标签 0=Fake, 1=Real。
    """
    if not isinstance(label_list, np.ndarray): label_list = np.array(label_list)
    if not isinstance(pred_probs, np.ndarray): pred_probs = np.array(pred_probs)

    if not label_list.size or not pred_probs.size:
        logger.warning("calculate_metrics: 标签列表或预测概率列表为空。")
        return {}
    if len(label_list) != len(pred_probs):
        logger.error(f"calculate_metrics: label_list ({len(label_list)}) 和 pred_probs ({len(pred_probs)}) 长度不匹配！")
        return {}

    # --- 计算总体指标 ---
    pred_labels = (pred_probs >= 0.5).astype(int) # 二分类阈值判断
    metrics = {}
    metrics['acc'] = accuracy_score(label_list, pred_labels)
    # 总体 Precision, Recall, F1 通常计算的是正类 (Real=1) 的指标
    metrics['precision'] = precision_score(label_list, pred_labels, pos_label=1, zero_division=0)
    metrics['recall'] = recall_score(label_list, pred_labels, pos_label=1, zero_division=0)
    metrics['F1'] = f1_score(label_list, pred_labels, pos_label=1, zero_division=0)
    try:
        # AUC 需要数据中同时包含 0 和 1 两类标签
        if len(np.unique(label_list)) > 1:
            metrics['auc'] = roc_auc_score(label_list, pred_probs)
        else:
            logger.warning(f"calculate_metrics: 数据中只存在单一类别标签，无法计算 AUC。")
            metrics['auc'] = 0.0 # 或者 None 或其他指示值
    except ValueError as e:
        logger.warning(f"计算 AUC 时出错: {e}")
        metrics['auc'] = 0.0

    # --- !!! 关键修改：计算 Real 和 Fake 类别的指标 !!! ---
    # Real 类 (标签=1)
    real_mask = (label_list == 1)
    if np.any(real_mask): # 只有存在 Real 样本时才计算
        metrics['Real'] = {
            'precision': precision_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'recall': recall_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'F1': f1_score(label_list, pred_labels, pos_label=1, zero_division=0),
            'support': int(np.sum(real_mask)) # 样本数量
        }
    else:
        metrics['Real'] = {'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'support': 0} # 或者空字典 {}

    # Fake 类 (标签=0)
    fake_mask = (label_list == 0)
    if np.any(fake_mask): # 只有存在 Fake 样本时才计算
        metrics['Fake'] = {
            # 注意：计算 Fake 类的指标时，pos_label 要设为 0
            'precision': precision_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'recall': recall_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'F1': f1_score(label_list, pred_labels, pos_label=0, zero_division=0),
            'support': int(np.sum(fake_mask)) # 样本数量
        }
    else:
        metrics['Fake'] = {'precision': 0.0, 'recall': 0.0, 'F1': 0.0, 'support': 0} # 或者空字典 {}

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
                  cat_metrics['precision'] = precision_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0) # 假设类别分析也关注正类1
                  cat_metrics['recall'] = recall_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0)
                  cat_metrics['F1'] = f1_score(cat_labels, cat_pred_labels, pos_label=1, zero_division=0)
                  try:
                      if len(np.unique(cat_labels)) > 1: cat_metrics['auc'] = roc_auc_score(cat_labels, cat_pred_probs)
                      else: cat_metrics['auc'] = 0.0
                  except ValueError: cat_metrics['auc'] = 0.0
                  metrics[category_name] = cat_metrics # 例如 metrics['gossip'] = {...}

    return metrics