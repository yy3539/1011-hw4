import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig,T5Tokenizer
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IGNORE_INDEX = -100 
TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
PAD_IDX = TOKENIZER.pad_token_id  # 更稳妥：用 tokenizer 的 pad_token_id

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--unfreeze_last_n_encoder_layers', type=int, default=None,
                    help="Number of last encoder+decoder layers to unfreeze for finetuning. None = unfreeze all")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']  # (B, T, V)

        # ====== 不再手工索引，用标准 flatten+ignore_index=-100 的写法 ======
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.view(-1, vocab_size),      # (B*T, V)
            decoder_targets.view(-1)          # (B*T,)
        )
        # ===============================================================

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            # 统计真正参与 loss 的 token 数（label != IGNORE_INDEX）
            valid = decoder_targets != IGNORE_INDEX
            num_tokens = valid.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    all_generated_sqls = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']  # (B, T, V)

            vocab_size = outputs.size(-1)
            loss = criterion(
                outputs.view(-1, vocab_size),
                decoder_targets.view(-1)
            )

            valid = decoder_targets != IGNORE_INDEX
            total_loss += loss.item() * valid.sum().item()
            total_tokens += valid.sum().item()

            # 下面的 generate / metrics 不用改
            generated_ids = model.generate(
                input_ids=encoder_input,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
                repetition_penalty=1.4,
            )
            for seq in generated_ids:
                sql = TOKENIZER.decode(seq, skip_special_tokens=True)
                sql = trim_generated_sql(sql) 
                all_generated_sqls.append(sql)

    # Save generated SQLs and compute their records
    save_queries_and_records(all_generated_sqls, model_sql_path, model_record_path)

    # Compute metrics
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_query_records=gt_record_path, model_query_records=model_record_path
    )
    error_rate = sum([1 for msg in model_error_msgs if msg]) / len(model_error_msgs)
    analyze_error_types(model_error_msgs)
    show_other_examples(model_error_msgs, n=20)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, record_em, record_f1, sql_em, error_rate

def trim_generated_sql(sql):
    # Take only the first SQL statement
    if ';' in sql:
        return sql.split(';')[0]
    return sql

from collections import Counter
def show_other_examples(model_error_msgs, n=10):
    for msg in [m for m in model_error_msgs if m]:
        lower = msg.lower()
        if not any(k in lower for k in ["syntax", "parse", "unexpected",
                                        "execution", "unknown", "no such",
                                        "column", "table",
                                        "mismatch", "type", "invalid", "ambiguous","incomplete","unrecognized"]):
            print(msg)
            n -= 1
            if n == 0:
                break
def analyze_error_types(model_error_msgs):
    # Filter out no-error entries
    errors = [msg for msg in model_error_msgs if msg]

    # Define categories and simple keyword matches
    categories = {
        "SyntaxError": ["syntax", "parse", "unexpected"],
        "ExecutionError": ["execution", "unknown", "no such", "column", "table"],
        "IncompleteError": ["incomplete"],
        "UnrecognizedToken": ["unrecognized"]
    }

    # Initialize counts
    counts = Counter({cat: 0 for cat in categories})

    # Categorize each message
    for msg in errors:
        lower = msg.lower()
        matched = False
        for cat, keywords in categories.items():
            if any(k in lower for k in keywords):
                counts[cat] += 1
                matched = True
                break
        if not matched:
            counts["Other"] += 1

    total = len(model_error_msgs)
    error_total = len(errors)
    error_rate = error_total / total if total else 0.0

    print(f"Total samples: {total}")
    print(f"Total error rate: {error_rate:.2%}\n")
    print("Error breakdown by type:")
    for cat, cnt in counts.items():
        print(f"  {cat:<15}: {cnt} ({cnt/total:.2%})")


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Inference loop for the test set. Saves generated SQL and records.
    """
    model.eval()
    all_generated_sqls = []

    with torch.no_grad():
        for encoder_input, encoder_mask,_ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # Generate SQL using greedy decoding
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                do_sample = False,
                num_beams = 1,
                early_stopping = False,
                repetition_penalty=1.4,
            )
            for seq in generated_ids:
                sql = TOKENIZER.decode(seq, skip_special_tokens=True)
                sql = trim_generated_sql(sql) 
                all_generated_sqls.append(sql)

    # Save SQL and records
    save_queries_and_records(all_generated_sqls, model_sql_path, model_record_path)
    print(f"Test inference done. SQL queries saved to {model_sql_path}, records saved to {model_record_path}")

class Args:
    # ----- Stage2: 只训练 decoder -----
    finetune = True
    stage = 2          # ★★★ 关键：切换到 stage 2
    freeze_decoder = False   # 这个字段在 stage 2 分支里其实不用，但你可以设成 False 以防以后用到

    # Stage2 我们只训 decoder，所以 unfreeze_last_n_encoder_layers 在目前代码里不会再用到
    unfreeze_last_n_encoder_layers = None

    optimizer_type = "AdamW"
    learning_rate = 1e-4      # ★★★ 建议用小一点的 lr，之前你就是这么设的
    weight_decay = 0.01

    scheduler_type = "linear"  # 你愿意的话可以改成 "cosine"
    num_warmup_epochs = 1
    max_n_epochs = 60          # Stage2 可以多训一点
    patience_epochs = 5        # 早停

    batch_size = 16
    test_batch_size = 16

    use_wandb = False
    experiment_name = "hw4_q7_stage2"  # ★★★ 换个 experiment 名，避免覆盖 stage1 的 ckpt

    # ★★★ 从 Stage1 的 best_model 加载权重
    resume_from_checkpoint = "checkpoints/ft_experiments/hw4_q7_stage1/best_model.pt"



def main():
    # Get key arguments
    args = Args()
    if args.use_wandb:
        setup_wandb(args)

    # Load the data
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

        # 初始化模型（会根据 args.resume_from_checkpoint 加载 Stage1 权重）
    model = initialize_model(args)

    # ===== 根据 stage 决定训练哪些部分 =====
    if getattr(args, "stage", 1) == 1 and getattr(args, "freeze_decoder", True):
        # ---------- Stage1: 只训练 encoder ----------
        print(">> Stage 1: encoder-only finetuning, freezing decoder and lm_head")
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        # encoder 的 requires_grad 逻辑在 initialize_model 里已经设好

    elif getattr(args, "stage", 1) == 2:
        # ---------- Stage2: 只训练 decoder ----------
        print(">> Stage 2: decoder-only finetuning, freezing encoder")

        # 先全部冻结，避免漏网之鱼
        for p in model.parameters():
            p.requires_grad = False

        # 解冻 decoder + lm_head
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True

        # 【可选】shared embedding 要不要训？
        # 如果你想“严格只动 decoder block”，可以保持 shared 冻结；
        # 如果你希望 decoder 的输入 embedding 也一起学习，可以把下面这段打开：
        #
        # for p in model.shared.parameters():
        #     p.requires_grad = True

    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    # ================================================

    # 再根据当前 requires_grad 设置 optimizer
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))


    # ===================== Train =====================
    print("Start training...")
    train(args, model, train_loader, dev_loader, optimizer, scheduler)
    print("Training finished.")

    # ===================== Evaluate (load best checkpoint) =====================
    # 训练过程中已经保存了 best_model.pt，这里重新加载最优模型再评估
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join('data/dev.sql')
    gt_record_path = os.path.join('records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')

    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, "
          f"Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
