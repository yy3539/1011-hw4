import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

def initialize_model(args):
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model.resize_token_embeddings(len(tokenizer))

    # 可选：从某个 checkpoint 恢复权重
    if getattr(args, "resume_from_checkpoint", None):
        print(f"Loading model weights from: {args.resume_from_checkpoint}")
        state_dict = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        model.load_state_dict(state_dict)

    # 先统一全部冻结，后面按 stage 决定解冻哪些部分
    for p in model.parameters():
        p.requires_grad = False

    stage = getattr(args, "stage", 1)   # 默认 stage1，防止没设出错

    # ==================== Stage 1: 只训练 encoder（freeze decoder） ====================
    # 你之前说的：“stage1: freeze decoder, lr=0.01”
    if stage == 1:
        # 解冻 encoder：全部或者最后 N 层
        n = getattr(args, "unfreeze_last_n_encoder_layers", None)
        if n is None:
            # 全部 encoder 都训练
            for p in model.encoder.parameters():
                p.requires_grad = True
        else:
            # 只训练 encoder 的最后 n 层
            blocks = getattr(model.encoder, "block", [])
            for layer in blocks[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # shared embedding（encoder/decoder 共享的词向量）
        # 如果希望 encoder 侧也能学，就解冻；decoder 的输出层依然是冻结的 lm_head
        for p in model.shared.parameters():
            p.requires_grad = True

        # decoder 和 lm_head 保持冻结（前面已经置 False，就不用再写）

    # ==================== Stage 2: 训练 decoder（也可以顺带 encoder） ====================
    elif stage == 2:
        # encoder 策略：沿用 unfreeze_last_n_encoder_layers 逻辑
        n = getattr(args, "unfreeze_last_n_encoder_layers", None)
        if n is None:
            # 想在 stage2 把 encoder 全部也训练起来：
            for p in model.encoder.parameters():
                p.requires_grad = True
        else:
            # 只训练 encoder 的最后 n 层（其余继续冻结）
            blocks = getattr(model.encoder, "block", [])
            for layer in blocks[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # decoder & lm_head 在 stage2 打开
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True

        # shared embedding 一般也一起训练
        for p in model.shared.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown stage: {stage}, should be 1 or 2")

    return model




def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best=False):
    # Save model checkpoint to be able to load the model later
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = "best_model.pt" if best else "last_model.pt"
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model {'best' if best else 'last'} checkpoint saved at {save_path}")
    
def load_model_from_checkpoint(args, best):
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    filename = "best_model.pt" if best else "last_model.pt"
    load_path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    # ⚠️ 关键：用和训练时完全一样的方式构建模型
    # （会用 "t5-small" + T5Tokenizer + resize_token_embeddings）
    from t5_utils import initialize_model, DEVICE  # 你已经在顶部 import 了就不需要再写这一行

    model = initialize_model(args)  # 这里不会再自动加载 checkpoint，因为你现在是手动 load_state_dict
    state_dict = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    print(f"Loaded {'best' if best else 'last'} model from {load_path}")
    return model

    
def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

