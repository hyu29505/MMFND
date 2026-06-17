# import torch, os, re, warnings
# import pandas as pd
# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
#     BitsAndBytesConfig
# )

# warnings.filterwarnings("ignore", message=".*attention mask.*")

# # 提取新浪图片URL中的图片名称
# def extract_sina_image_name(url):
#     # 1. 去除URL末尾可能的|null等占位符
#     pure_url = url.split('|')[0]
#     # 2. 按“large/”分割，取后面的部分（即图片文件名）
#     if 'large/' in pure_url:
#         image_name = pure_url.split('large/')[1]
#         # 3. 确保提取结果是.jpg文件（过滤异常链接）
#         if image_name.endswith('.jpg'):
#             return image_name
#     return "无效图片URL"

# MODEL_PATH = "./Qwen-VL"

# # 1. 8-bit 量化
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# print("Loading Qwen-VL-7B (8-bit)...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     device_map="auto",
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     # ❗ 先去掉 Flash-Attn：官方未支持
#     # attn_implementation="flash_attention_2"
# )

# # 2. 限制生成长度
# model.generation_config = GenerationConfig.from_pretrained(
#     MODEL_PATH, trust_remote_code=True, max_new_tokens=256
# )

# # ---------- 3. 数据 ----------
# DATA_DIR = "./data"
# TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
# TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
# NONRUMOR_IMG_DIR = os.path.join(DATA_DIR, "nonrumor_images")  # 真新闻图片目录
# RUMOR_IMG_DIR    = os.path.join(DATA_DIR, "rumor_images")     # 虚假新闻图片目录

# # 读取并合并训练集和测试集
# df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
# print(f"Total samples: {len(df)}")

# samples = []
# for _, r in df.iterrows():
#     # 获取图片URL并提取图片名称
#     img_url = str(r["image_url"]) if pd.notna(r["image_url"]) else "null"
#     img_id = extract_sina_image_name(img_url) if img_url.lower() != "null" else "null"
    
#     img_path = None
#     if img_id != "无效图片URL" and img_id.lower() != "null":
#         # 根据label确定图片目录：0是虚假新闻对应rumor_images，1是真新闻对应nonrumor_images
#         if r["label"] == 0:  # 虚假新闻
#             img_path = os.path.join(RUMOR_IMG_DIR, img_id)
#         else:  # 真实新闻
#             img_path = os.path.join(NONRUMOR_IMG_DIR, img_id)
    
#     samples.append({
#         "post_id": r["post_id"],
#         "image_id": img_url,
#         "image_id": img_id,
#         "label": r["label"],
#         "category": r["category"],
#         "content": r["content"],
#         "image_path": img_path,
#     })

# # ---------- 4. 合并 prompt ----------
# PROMPT_ALL = (
#     "请一次性完成下面三个子任务，并以换行+===+换行分隔三段输出：\n"
#     "1) 分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图？\n"
#     "2) 检查此图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图？\n"
#     "3) 对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图？\n\n"
#     "文本：{content}"
# )

# results = []
# for item in tqdm(samples, desc="Generating"):
#     content, img_path = item["content"], item["image_path"]
#     prompt = PROMPT_ALL.format(content=content)

#     if img_path and os.path.exists(img_path):
#         query = [{"image": img_path}, {"text": prompt}]
#     else:
#         query = prompt  # 无图或图片不存在

#     resp, _ = model.chat(tokenizer, query=query, history=None)

#     # 切三段
#     parts = [p.strip() for p in re.split(r"\n===\n", resp) if p.strip()]
#     while len(parts) < 3:
#         parts.append("")
#     txt_r, img_r, cross_r = parts

#     results.append({
#         **item,
#         "text_reasoning": txt_r,
#         "image_reasoning": img_r,
#         "cross_modal_reasoning": cross_r,
#     })

# # ---------- 5. 保存 ----------
# out_csv = "./weibo_with_reasoning.csv"
# pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
# print(f"Done → {out_csv}")
import torch, os, re, warnings
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    BitsAndBytesConfig
)

warnings.filterwarnings("ignore", message=".*attention mask.*")

# 提取新浪图片URL中的图片名称，支持多个URL
def extract_sina_image_names(url):
    # 1. 按|分割多个URL
    urls = url.split('|') if url and isinstance(url, str) else []
    image_names = []
    
    for pure_url in urls:
        # 过滤空值和null
        if not pure_url or pure_url.lower() == "null":
            continue
            
        # 2. 按“large/”分割，取后面的部分（即图片文件名）
        if 'large/' in pure_url:
            image_name = pure_url.split('large/')[1]
            # 3. 确保提取结果是.jpg文件（过滤异常链接）
            if image_name.endswith('.jpg'):
                image_names.append(image_name)
    
    return image_names if image_names else ["无效图片URL"]

def find_image_column(df):
    """更灵活地查找可能的图片列"""
    possible_image_columns = [
        'image_url', 'img_url', 'images', 'image', 
        'pic_url', 'picture_url', 'image_links'
    ]
    for col in possible_image_columns:
        if col in df.columns:
            return col
    
    # 检查是否有包含"image"或"img"的列
    for col in df.columns:
        if 'image' in col.lower() or 'img' in col.lower():
            return col
    
    return None

MODEL_PATH = "./Qwen-VL"

# 1. 8-bit 量化
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading Qwen-VL-7B (8-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    # 保持原配置，不使用Flash-Attn
)

# 2. 限制生成长度
model.generation_config = GenerationConfig.from_pretrained(
    MODEL_PATH, trust_remote_code=True, max_new_tokens=256
)

# ---------- 3. 数据 ----------
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
NONRUMOR_IMG_DIR = os.path.join(DATA_DIR, "nonrumor_images")  # 真新闻图片目录
RUMOR_IMG_DIR    = os.path.join(DATA_DIR, "rumor_images")     # 虚假新闻图片目录

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 读取并合并训练集和测试集
try:
    df_train = pd.read_csv(TRAIN_CSV) if os.path.exists(TRAIN_CSV) else pd.DataFrame()
    df_test = pd.read_csv(TEST_CSV) if os.path.exists(TEST_CSV) else pd.DataFrame()
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total samples: {len(df)}")
    
    if len(df) == 0:
        print("警告：训练集和测试集都是空的，请检查数据文件路径")
except Exception as e:
    print(f"读取数据时出错: {str(e)}")
    exit(1)

# 检查并获取图片URL的列名
image_column = find_image_column(df)

if image_column is None:
    print(f"警告：未找到图片URL列，数据集中的列名为：{df.columns.tolist()}")
    print("将继续处理文本数据，忽略图片相关分析")
else:
    print(f"找到图片URL列: {image_column}")

samples = []
for _, r in df.iterrows():
    # 初始化图片相关变量
    img_url = "null"
    img_ids = ["null"]
    img_paths = []
    
    # 只有当找到图片列时才处理图片
    if image_column is not None and image_column in df.columns:
        try:
            img_url = str(r[image_column]) if pd.notna(r[image_column]) else "null"
        except KeyError:
            img_url = "null"
            
        img_ids = extract_sina_image_names(img_url) if img_url.lower() != "null" else ["null"]
        
        # 根据label确定图片目录：0是虚假新闻对应rumor_images，1是真新闻对应nonrumor_images
        try:
            img_dir = RUMOR_IMG_DIR if r["label"] == 0 else NONRUMOR_IMG_DIR
            
            # 确保图片目录存在
            os.makedirs(img_dir, exist_ok=True)
            
            for img_id in img_ids:
                if img_id != "无效图片URL" and img_id.lower() != "null":
                    img_path = os.path.join(img_dir, img_id)
                    img_paths.append(img_path)
        except Exception as e:
            print(f"处理图片路径时出错: {str(e)}")
            img_paths = []
    
    # 确保content字段存在且不为空
    content = str(r.get("content", "")) if pd.notna(r.get("content")) else str(r.get("post_text", ""))
    
    samples.append({
        "post_id": r.get("post_id", ""),
        "image_url": img_url,
        "image_ids": img_ids,
        "label": r.get("label", ""),
        "category": r.get("category", ""),
        "content": content,
        "image_paths": img_paths,
    })

# ---------- 4. 合并 prompt ----------
# 根据是否有图片调整提示词
if image_column is None:
    PROMPT_ALL = (
        "请一次性完成下面三个子任务，并以换行+===+换行分隔三段输出：\n"
        "1) 分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图？\n"
        "2) （无图片可用）\n"
        "3) （无图片可用，无法进行文本与图像的对比分析）\n\n"
        "文本：{content}"
    )
else:
    PROMPT_ALL = (
        "请一次性完成下面三个子任务，并以换行+===+换行分隔三段输出：\n"
        "1) 分析此文本，是否存在误导性语言、情感操纵或逻辑谬误？请逐步推理其虚假的可能性及操纵意图？\n"
        "2) 检查提供的图像，是否存在编辑痕迹、不合情理的元素或与常识相悖的场景？请逐步推理其虚假的可能性及操纵意图？\n"
        "3) 对比文本和图像，它们之间是否存在矛盾、不协调或刻意营造的虚假关联？请逐步推理其虚假的可能性及操纵意图？\n\n"
        "文本：{content}"
    )

results = []
for item in tqdm(samples, desc="Generating"):
    content, img_paths = item["content"], item["image_paths"]
    prompt = PROMPT_ALL.format(content=content)

    # 构建查询
    try:
        # 检查图片路径是否有效
        valid_img_paths = [p for p in img_paths if os.path.exists(p) and os.path.getsize(p) > 0]
        
        if valid_img_paths:
            query = []
            # 添加所有有效图片
            for img_path in valid_img_paths:
                query.append({"image": img_path})
            # 添加文本提示
            query.append({"text": prompt})
        else:
            # 无有效图片，仅使用文本
            if image_column is None:
                query = prompt + "\n提示：该数据无关联图片，仅基于文本进行分析。"
            else:
                query = prompt + "\n提示：未找到有效图片文件，无法进行图像分析。"

        resp, _ = model.chat(tokenizer, query=query, history=None)

        # 切三段
        parts = [p.strip() for p in re.split(r"\n===\n", resp) if p.strip()]
        while len(parts) < 3:
            parts.append("")
        txt_r, img_r, cross_r = parts[:3]  # 确保只取前三部分

        results.append({
            **item,
            "text_reasoning": txt_r,
            "image_reasoning": img_r,
            "cross_modal_reasoning": cross_r,
        })
    except Exception as e:
        print(f"处理样本 {item.get('post_id', '未知ID')} 时出错: {str(e)}")
        # 出错时添加空结果，保证数据完整性
        results.append({
            **item,
            "text_reasoning": f"处理出错: {str(e)}",
            "image_reasoning": "",
            "cross_modal_reasoning": "",
        })

# ---------- 5. 保存 ----------
out_csv = "./weibo_with_reasoning.csv"
try:
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"完成！结果已保存至 → {out_csv}")
except Exception as e:
    print(f"保存结果时出错: {str(e)}")
    # 尝试保存到备用路径
    out_csv = "./weibo_with_reasoning_backup.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已尝试保存到备用路径 → {out_csv}")
