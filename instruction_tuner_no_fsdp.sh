DATA_NAME_OR_PATH="kowndinya23/tulu-v2-sft-mixture-150K"
DATASET_LABEL="tulu-v2-sft-mixture-150K"
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B"
MODEL_LABEL="llama-3-3b"
LEARNING_RATE=2e-5
NUM_EPOCHS=1
CONFIG_FILE_NAME=config_no_fsdp.yaml
TRAIN_FILE=instruction_tuner_wit_loss_no_fsdp.py

LAMBDA_P=0.4
LAMBDA_R=0.6

OUTPUT_DIR=$DATASET_LABEL-$MODEL_LABEL-$NUM_EPOCHS-epochs-lambda_p-$LAMBDA_P-lambda_r-$LAMBDA_R
HUB_TOKEN=YOUR_TOKEN_HERE

export WANDB_API_KEY=YOUR_TOKEN_HERE
wandb login --cloud --host https://api.wandb.ai --relogin $WANDB_API_KEY


accelerate launch --config_file $CONFIG_FILE_NAME $TRAIN_FILE \
    --dataset_name_or_path $DATA_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attention_2 \
    --hf_access_token $HUB_TOKEN \
    --torch_dtype bfloat16 \
    --max_seq_length 4096 \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --preprocessing_num_workers 12 \
    --seed 23 \
    --num_train_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps 8 \
    --prompt_tokens_weight $LAMBDA_P \
    --response_tokens_weight $LAMBDA_R \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --lr_warmup_fraction 0.01 \
    --with_tracking \
    --report_to wandb \
    --output_dir $OUTPUT_DIR \
    --push_to_hub \
    --hub_token $HUB_TOKEN \
    --private_repo