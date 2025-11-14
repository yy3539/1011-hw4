# load_data.py
import os, re, json, platform, random, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
# 原模板已有 punkt，这里静默确保 wordnet 也可用（失败不致命）
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    try:
        nltk.download('wordnet', quiet=True)
    except Exception:
        pass

from nltk.stem import WordNetLemmatizer

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0  # T5 的 pad_token_id == 0


# ----------------------------
# Schema 解析 & 格式化
# ----------------------------
def parse_schema_to_dict(schema_path="data/flight_database.schema"):
    """
    支持两种格式：
    1) JSON（包含 ents -> {table: {col: {...}}}）
    2) 旧的行文本：table(col1, col2, ...)
    返回: {table: {col: "col", ...}, ...}
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # 优先尝试 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "ents" in obj and isinstance(obj["ents"], dict):
            out = {}
            for table, cols_dict in obj["ents"].items():
                if isinstance(cols_dict, dict):
                    out[table] = {col: "col" for col in cols_dict.keys()}
            if out:
                return out
    except Exception:
        pass  # 不是 JSON，回退到行解析

    # 回退：解析形如 table(col1, col2, ...)
    schema_dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", line)
        if not m:
            continue
        table = m.group(1)
        cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
        if cols:
            schema_dict[table] = {c.split()[0]: "col" for c in cols}
    return schema_dict


def schema_to_prompt_subset(subset_dict):
    """
    把 {table: {col: type}} 转成多行字符串（只保留表:列）：
    table_a: col1, col2, ...
    table_b: col1, col2, ...
    """
    lines = []
    for table, cols in subset_dict.items():
        col_list = ", ".join(sorted(cols.keys()))
        lines.append(f"{table}: {col_list}")
    return "\n".join(lines)


# ----------------------------
# NL → 相关表筛选（可选）
# ----------------------------
lemmatizer = WordNetLemmatizer()

def get_relevant_tables(nl_query, schema_dict, tokenizer, max_model_tokens):
    """
    基于非常轻量的词/词形重叠对表/列打分，并在 token 预算内选择若干表及其“更相关”的列。
    max_model_tokens 为 schema 文本预算（不含 prompt 其它部分）。
    """
    if max_model_tokens <= 0:
        return {}

    nl_words = re.findall(r"\w+", nl_query.lower())
    try:
        nl_lemmas = set(lemmatizer.lemmatize(w) for w in nl_words)
    except Exception:
        nl_lemmas = set(nl_words)

    # 表打分 + 每表列打分
    table_scores = []
    per_table_col_scores = {}
    for table, cols in schema_dict.items():
        table_words = table.lower().split("_")
        try:
            table_lemmas = set(lemmatizer.lemmatize(w) for w in table_words)
        except Exception:
            table_lemmas = set(table_words)

        tscore = 1.5 * sum(1 for w in table_words if w in nl_words) + 1.0 * len(table_lemmas & nl_lemmas)

        col_scored = []
        for col in cols.keys():
            col_words = col.lower().split("_")
            try:
                col_lemmas = set(lemmatizer.lemmatize(w) for w in col_words)
            except Exception:
                col_lemmas = set(col_words)
            cscore = 0.5 * sum(1 for w in col_words if w in nl_words) + 0.5 * len(col_lemmas & nl_lemmas)
            col_scored.append((cscore, col))
            tscore += 0.05 * cscore  # 列相关性给表一个微弱加成，利于排序

        col_scored.sort(reverse=True, key=lambda x: x[0])
        per_table_col_scores[table] = col_scored
        if tscore > 0:
            table_scores.append((tscore, table))

    table_scores.sort(reverse=True, key=lambda x: x[0])

    selected = {}
    tokens_used = 0
    for _, table in table_scores:
        cols_sorted = [c for s, c in per_table_col_scores[table] if s > 0] or list(schema_dict[table].keys())

        current_cols = []
        for col in cols_sorted:
            trial_cols = current_cols + [col]
            schema_text = schema_to_prompt_subset({table: {c: "col" for c in trial_cols}})
            need = len(tokenizer.tokenize(schema_text))
            if tokens_used + need <= max_model_tokens:
                current_cols = trial_cols
            else:
                break

        if not current_cols:
            # 兜底：尝试整表
            schema_text_full = schema_to_prompt_subset({table: schema_dict[table]})
            need_full = len(tokenizer.tokenize(schema_text_full))
            if tokens_used + need_full <= max_model_tokens:
                selected[table] = schema_dict[table]
                tokens_used += need_full
        else:
            selected[table] = {c: "col" for c in current_cols}
            tokens_used += len(tokenizer.tokenize(schema_to_prompt_subset({table: selected[table]})))

    return selected


# ----------------------------
# Prompt 模板
# ----------------------------
PROMPT_TEMPLATE = """NL Query: "{query}"

Relevant schema:
{schema}

SQL:
"""


# ----------------------------
# 读取辅助
# ----------------------------
def load_lines(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


# ----------------------------
# Dataset
# ----------------------------
class T5Dataset(Dataset):
    def __init__(self, data_folder, split, encoder_max_len=512, decoder_max_len=128, use_relevant_schema=True, schema_budget=320):
        """
        data_folder: 里应包含 train/dev/test 的 .nl 与 .sql（test 无 .sql）
        split: "train" | "dev" | "test"
        encoder_max_len / decoder_max_len: 截断长度
        use_relevant_schema: 是否按 NL 选择相关表
        schema_budget: 分配给 schema 文本的 token 预算（粗估），最终仍受 encoder_max_len 控制
        """
        self.split = split
        self.data_folder = data_folder
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.use_relevant_schema = use_relevant_schema
        self.schema_budget = schema_budget

        # tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        # T5 没有专用 BOS，常用做法是使用 pad_token（id=0）作为 decoder_start_token
        self.bos_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else PAD_IDX

        # schema
        schema_path = os.path.join(os.path.dirname(__file__), "data", "flight_database.schema")
        self.schema_dict = parse_schema_to_dict(schema_path)

        # process
        self.samples = self.process_data(self.data_folder, self.split, self.tokenizer)

    def clean_nl(self, text: str) -> str:
        return text.strip()

    def clean_sql(self, text: str) -> str:
        return text.strip()

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        sql_lines = None
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            assert len(nl_lines) == len(sql_lines), f"NL/SQL 数量不一致: {len(nl_lines)} vs {len(sql_lines)}"

        samples = []
        for i, nl in enumerate(nl_lines):
            question_text = self.clean_nl(nl)

            # 计算 prompt 中非 schema 部分占用，以便给 schema 预留预算
            prefix_text = PROMPT_TEMPLATE.format(query=question_text, schema="")
            prefix_ids = tokenizer(
                prefix_text,
                truncation=True,
                max_length=self.encoder_max_len,
                add_special_tokens=True,
                return_attention_mask=False
            )["input_ids"]
            room_for_schema = max(0, self.encoder_max_len - len(prefix_ids))
            # 再和 schema_budget 取较小值，避免 schema 膨胀
            room_for_schema = min(room_for_schema, self.schema_budget)

            # 选择 schema 子集
            if self.use_relevant_schema:
                subset = get_relevant_tables(question_text, self.schema_dict, tokenizer, max_model_tokens=room_for_schema)
                if not subset:
                    subset = self.schema_dict  # 兜底：放全部
            else:
                subset = self.schema_dict

            schema_text = schema_to_prompt_subset(subset)

            enc_text = PROMPT_TEMPLATE.format(
                query=question_text,
                schema=schema_text,
            )
            enc = tokenizer(
                enc_text,
                truncation=True,
                max_length=self.encoder_max_len,
                return_attention_mask=True,
                add_special_tokens=True,
            )
            enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            enc_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

            if split != "test":
                sql = self.clean_sql(sql_lines[i])
                dec_ids = tokenizer(
                    sql,
                    truncation=True,
                    max_length=self.decoder_max_len,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
                dec_ids = torch.tensor(dec_ids, dtype=torch.long)
            else:
                dec_ids = None

            samples.append({
                "enc_ids": enc_ids,
                "enc_mask": enc_mask,
                "dec_ids": dec_ids,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ----------------------------
# Collate functions
# ----------------------------
def _pad_1d(batch_tensors, pad_value=PAD_IDX):
    return pad_sequence(batch_tensors, batch_first=True, padding_value=pad_value)

def normal_collate_fn(batch):
    """
    训练/验证 collate：返回 5 个对象
    encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
    """
    enc_ids = [b["enc_ids"] for b in batch]
    enc_mask = [b["enc_mask"] for b in batch]
    dec_targets = [b["dec_ids"] for b in batch]
    # 防御：确保没有 None（test 不会走这里）
    assert all(t is not None for t in dec_targets), "normal_collate_fn 不应在 test split 上使用"

    enc_ids_pad = _pad_1d(enc_ids, PAD_IDX)    # encoder PAD=0
    enc_mask_pad = _pad_1d(enc_mask, 0)        # mask PAD=0

    # T5 decoder 起始用 pad_token_id 作为 BOS
    BOS = PAD_IDX
    decoder_inputs, decoder_targets = [], []
    for tgt in dec_targets:
        # teacher forcing：把 gold 右移一位，并在最前面放 BOS
        inp = torch.cat([torch.tensor([BOS], dtype=torch.long), tgt[:-1]], dim=0)
        decoder_inputs.append(inp)
        decoder_targets.append(tgt)

    decoder_inputs_pad = _pad_1d(decoder_inputs, PAD_IDX)
    decoder_targets_pad = _pad_1d(decoder_targets, PAD_IDX)

    # labels 的 PAD 置为 -100，loss 忽略
    decoder_targets_pad = decoder_targets_pad.masked_fill(
        decoder_targets_pad == PAD_IDX, -100
    )

    # 评估用：decoder 初始喂一个 BOS
    initial_decoder_inputs = torch.full((len(batch), 1), fill_value=BOS, dtype=torch.long)

    return enc_ids_pad, enc_mask_pad, decoder_inputs_pad, decoder_targets_pad, initial_decoder_inputs


def test_collate_fn(batch):
    """
    推理 collate：返回 3 个对象
    encoder_ids, encoder_mask, initial_decoder_inputs
    """
    enc_ids = [b["enc_ids"] for b in batch]
    enc_mask = [b["enc_mask"] for b in batch]
    enc_ids_pad = _pad_1d(enc_ids, PAD_IDX)
    enc_mask_pad = _pad_1d(enc_mask, 0)
    BOS = PAD_IDX
    initial_decoder_inputs = torch.full((len(batch), 1), fill_value=BOS, dtype=torch.long)
    return enc_ids_pad, enc_mask_pad, initial_decoder_inputs


# ----------------------------
# DataLoader 构造
# ----------------------------
def get_dataloader(batch_size, split, encoder_max_len=512, decoder_max_len=128, num_workers=None):
    data_folder = 'data'
    dset = T5Dataset(
        data_folder, split,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len
    )
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    # 更稳的 DataLoader 配置（Windows 关 persistent_workers）
    is_windows = platform.system().lower().startswith("win")
    auto_workers = min(4, os.cpu_count() or 1)
    num_workers = auto_workers if num_workers is None else num_workers
    pin_memory = torch.cuda.is_available()
    persistent_workers = (not is_windows) and (num_workers > 0)

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers and num_workers > 0 else None,
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size, encoder_max_len=512, decoder_max_len=128, eval_num_workers=0):
    train_loader = get_dataloader(
        batch_size, "train",
        encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len,
        num_workers=None  # 训练阶段用自动选择
    )
    dev_loader = get_dataloader(
        test_batch_size, "dev",
        encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len,
        num_workers=eval_num_workers
    )
    test_loader = get_dataloader(
        test_batch_size, "test",
        encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len,
        num_workers=eval_num_workers
    )
    return train_loader, dev_loader, test_loader


# ----------------------------
# Prompting 数据（原 TODO）
# ----------------------------
def load_prompting_data(data_folder):
    """
    简单读入原始 NL / SQL 文本，便于构造 few-shot / in-context 数据。
    返回:
        train_x, train_y, dev_x, dev_y, test_x
        其中 *_x 为 NL 列表，*_y 为 SQL 列表（test 没有 y）
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
