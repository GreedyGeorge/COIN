

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 30124 --multi_gpu train_corrector.py \
  --do_train \
  --do_eval \
  --train_on lawtrain.txt \
  --eval_on law_pr_test.json \
  --output_dir models/coin_law\
  --max_train_steps 10000 \
  --fp16 \
  --model_type coin \
  --mft \
  --gradient_accumulation_steps 1 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --save_steps 100 \
  --learning_rate 5e-5 \
  --max_seq_length 128

