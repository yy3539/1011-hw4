import torch, os
from transformers import AutoTokenizer, GenerationConfig
from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from train_t5 import eval_epoch, DEVICE

args = type("Args", (), {})()
args.finetune = True
args.freeze_encoder = True
args.eval_num_workers = 0
args.eval_num_beams = 4
args.eval_max_new_tokens = 256
args.min_new_tokens = 32
args.length_penalty = 0.8
args.no_repeat_ngram_size = 4
args.eval_length_penalty = 0.8
args.eval_no_repeat_ngram_size = 4
args.encoder_max_len = 512
args.decoder_max_len = 256
args.experiment_name = "hw4_q7_stage2_dec_tune_resume"
args.load_from_checkpoint = "checkpoints/ft_experiments/hw4_q7_stage2_dec_tune_resume/best.pt"

train_loader, dev_loader, test_loader = load_t5_data(
    batch_size=16, test_batch_size=16,
    encoder_max_len=args.encoder_max_len, decoder_max_len=args.decoder_max_len,
    eval_num_workers=args.eval_num_workers
)

model = initialize_model(args)
state_dict = torch.load(args.load_from_checkpoint, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

gt_sql_path = os.path.join("data", "dev.sql")
gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
save_prefix = f"t5_ft_{args.experiment_name}"

eval_loss, record_f1, record_em, sql_em = eval_epoch(
    args, model, dev_loader, gt_sql_path, gt_record_path, save_prefix
)
print(f"[EVAL ONLY] Dev Loss={eval_loss:.4f}, Record F1={record_f1:.4f}, EM={record_em:.4f}, SQL EM={sql_em:.4f}")
