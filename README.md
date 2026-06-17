# MM: Multimodal Fake News Detection

This repo provides training entry points for multimodal fake news detection
across GossipCop, Weibo, and Weibo21. There is no single "main" script; use the
dataset-specific entry points below. The training logic is implemented in
`run.py`, and the CLI wrappers live in `main_weibo.py` and `main_gossip.py`.

## Quick Start

Install dependencies (from repo root `mm/`):

```bash
pip install -r ./requirements.txt
```

## Run: Weibo

```bash
python ./main_weibo.py \
  --dataset weibo \
  --weibo_data_dir ../data \
  --bert_model_path_weibo ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch \
  --bert_vocab_file_weibo ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt \
  --gpu 0
```

Expected Weibo files (under `--weibo_data_dir`):

- `train_origin.csv`
- `val_origin.csv`
- `test_origin.csv`
- `train_loader.pkl`, `train_clip_loader.pkl`
- `val_loader.pkl`, `val_clip_loader.pkl`
- `test_loader.pkl`, `test_clip_loader.pkl`

## Run: Weibo21

```bash
python ./main_weibo.py \
  --dataset weibo21 \
  --weibo21_data_dir ./Weibo_21 \
  --bert_model_path_weibo ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch \
  --bert_vocab_file_weibo ./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt \
  --gpu 0
```

Expected Weibo21 files (under `--weibo21_data_dir`):

- `train_datasets.xlsx`
- `val_datasets.xlsx`
- `test_datasets.xlsx`
- `train_loader.pkl`, `train_clip_loader.pkl`
- `val_loader.pkl`, `val_clip_loader.pkl`
- `test_loader.pkl`, `test_clip_loader.pkl`

## Run: GossipCop

```bash
python ./main_gossip.py \
  --dataset gossipcop \
  --gossipcop_data_dir ../gossipcop \
  --bert_model_path_gossipcop ./pretrained_model/bert-base-uncased \
  --clip_model_path_gossipcop ./pretrained_model/clip-vit-base-patch16 \
  --gpu 0
```

## Notes

- Default hyperparameters are set in `main_weibo.py` and `main_gossip.py`.
- Model training and dataloaders are wired in `run.py`.
- Model checkpoints are saved under `--save_param_dir` (default:
  `./param_model`).

