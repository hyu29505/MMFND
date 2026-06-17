# main.py 
# -*-coding = utf-8 -*-

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# **********************************************************************************************************************************
# gossipcop weibo21 weibo需修改model_name dataset early_stop_metric batchsize lr seed  
#  每一个参数的 help choices 一定要看
# gossipcop lr  0.0003    bs  24   seed  2024    early_stop_metric    acc
# weibo     lr  0.000175  bs  32   seed  3074    early_stop_metric    F1
# weibo21   lr  0.0005    bs  64   seed  3074    early_stop_metric    F1

# ************************************************************************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='domain_weibo', help="要使用的模型架构名称。目前gossipcop使用'domain_gossipcop', weibo/weibo21使用'domain_weibo'")
parser.add_argument('--dataset', default='weibo21', choices=['gossipcop', 'weibo', 'weibo21'], help="要使用的数据集。")
parser.add_argument('--epoch', type=int, default=200, help="训练周期数。")
parser.add_argument('--max_len', type=int, default=197, help="BERT 的最大序列长度。")
parser.add_argument('--num_workers', type=int, default=4, help="dataloader 的 worker 数量。")
parser.add_argument('--early_stop', type=int, default=100, help="早停的耐心值。对于Weibo默认为100，GossipCop为100。") # 将根据dataset调整
parser.add_argument('--early_stop_metric', default='acc', choices=['acc', 'F1'], help="用于早停的指标 ('acc' 或 'F1')，acc主要用于GossipCop。Weibo weibo21使用'metric'（通常是F1）。")

# --- GossipCop 相关模型路径 ---
parser.add_argument('--bert_model_path_gossipcop', default='./pretrained_model/bert-base-uncased', help="GossipCop 使用的英文 BERT 模型本地路径。")
parser.add_argument('--clip_model_path_gossipcop', default='./pretrained_model/clip-vit-base-patch16', help="GossipCop 使用的英文 CLIP 模型本地路径。")

# --- Weibo/Weibo21 相关模型路径 ---
parser.add_argument('--bert_model_path_weibo', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch', help="Weibo/Weibo21 BERT 模型本地路径。")
parser.add_argument('--bert_vocab_file_weibo', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt', help="Weibo/Weibo21 BERT 词汇表文件。")
# Weibo/Weibo21的CLIP模型通常由cn_clip内部加载或指定路径，这里暂时不设独立参数，由domain_weibo.py内部处理cn_clip的加载。

# --- 数据集根目录 ---
parser.add_argument('--gossipcop_data_dir', default='./gossipcop/', help="GossipCop 数据集根目录。")
parser.add_argument('--weibo_data_dir', default='./data/', help="Weibo 数据集根目录。")
parser.add_argument('--weibo21_data_dir', default='./Weibo_21/', help="Weibo21 数据集根目录。")

parser.add_argument('--batchsize', type=int, default=32, help="批处理大小。Weibo 为32 weibo21默认为64，GossipCop为32。") # 将根据dataset调整
parser.add_argument('--seed', type=int, default=2024, help="随机种子。Weibo weibo21默认为3074，GossipCop为2024。") # 将根据dataset调整
parser.add_argument('--gpu', default='0', help="要使用的 GPU ID。")
parser.add_argument('--bert_emb_dim', type=int, default=768, help="BERT 嵌入维度。")
parser.add_argument('--lr', type=float, default=0.0005, help="学习率。GossipCop默认为0.0005。weibo21 0.0005 weibo 0.000175") # 将根据dataset调整
parser.add_argument('--emb_type', default='bert', help="嵌入类型 (主要为Weibo流程保留，GossipCop固定使用bert)。")
parser.add_argument('--save_param_dir', default= './param_model', help="保存模型参数的目录。")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 在设置 CUDA_VISIBLE_DEVICES 后导入 Run
try:
    from run import Run # 假设 run.py 是整合后的版本
except ImportError as e:
     logger.error(f"无法导入 Run 类: {e}. 请确保 run.py 文件存在且无误。")
     exit()

import torch
import numpy as np
import random

# --- 根据数据集调整部分默认参数 ---
if args.dataset == 'gossipcop':
    current_seed = args.seed if args.seed != 3074 else 2024 # 如果用户没特定指定，则用GossipCop默认
    current_batchsize = args.batchsize if args.batchsize != 64 else 24
    current_lr = args.lr if args.lr != 0.0001 else 0.0003
    current_early_stop = args.early_stop if args.early_stop != 6 else 10
    current_model_name = args.model_name if args.model_name != 'domain' else 'domain_gossipcop'
else: # weibo or weibo21
    current_seed = args.seed if args.seed != 2024 else 3074
    current_batchsize = args.batchsize if args.batchsize != 24 else 64
    current_lr = args.lr # 保持用户指定的或0.0001
    current_early_stop = args.early_stop if args.early_stop != 10 else 6
    current_model_name = args.model_name if args.model_name != 'domain' else 'domain_weibo'


# --- 设置种子 ---
random.seed(current_seed)
np.random.seed(current_seed)
torch.manual_seed(current_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Weibo main.py中包含 torch.backends.cudnn.enabled = True，通常默认即为True，除非有特定原因关闭

logger.info(f"使用数据集: {args.dataset}")
logger.info(f"使用模型: {current_model_name}")
logger.info(f"使用 GPU: {args.gpu}")
logger.info(f"种子设置为: {current_seed}")

emb_dim = args.bert_emb_dim # 主要使用 BERT 的维度

# --- 构建配置字典 ---
config = {
        'use_cuda': True if torch.cuda.is_available() else False,
        'dataset': args.dataset,
    'model_name': current_model_name, # 使用调整后的模型名称

        'gossipcop_data_dir': args.gossipcop_data_dir,
        'weibo_data_dir': args.weibo_data_dir,
        'weibo21_data_dir': args.weibo21_data_dir,

        'bert_model_path_gossipcop': args.bert_model_path_gossipcop,
        'clip_model_path_gossipcop': args.clip_model_path_gossipcop,
        'bert_model_path_weibo': args.bert_model_path_weibo,
        'bert_vocab_file_weibo': args.bert_vocab_file_weibo,
        # CLIP for Weibo (cn_clip) is usually handled by its own loading mechanism or a model_path string

        'batchsize': current_batchsize,
        'max_len': args.max_len,
        'early_stop': current_early_stop,
        'early_stop_metric': args.early_stop_metric, # GossipCop使用, Weibo trainer内部可能用固定值如'metric'
        'num_workers': args.num_workers,
        'emb_type': args.emb_type, # 主要为Weibo
        'weight_decay': 5e-5,
        'model_params': {'mlp': {'dims': [384], 'dropout': 0.2}}, # 通用结构，具体模型可能覆盖
        'emb_dim': emb_dim,
        'lr': current_lr,
        'epoch': args.epoch,
        'seed': current_seed,
        'save_param_dir': args.save_param_dir,
        }

# 为Weibo/Weibo21添加 vocab_file (与bert_vocab_file_weibo相同) 和 bert (与bert_model_path_weibo相同)
# 这是因为原始的Weibo run.py和domain.py中直接使用了这些键名
if args.dataset == 'weibo' or args.dataset == 'weibo21':
    config['vocab_file'] = args.bert_vocab_file_weibo # Weibo dataloader需要
    config['bert'] = args.bert_model_path_weibo       # Weibo domain model需要

logger.info(f"--- 完整配置 ---")
for key, value in config.items():
    logger.info(f"{key}: {value}")
logger.info(f"--------------------")

if __name__ == '__main__':
    if config['use_cuda']:
        logger.info(f"CUDA 可用。正在使用 GPU {args.gpu}。")
    else:
        logger.warning("CUDA 不可用。正在使用 CPU 运行。")

    runner = Run(config=config)
    runner.main()
    logger.info("运行结束。")


