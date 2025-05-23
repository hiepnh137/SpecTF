# export CUDA_VISIBLE_DEVICES=0
pipeline_name="ours"
# all_models=("iTransformer")
model_name="TimeMixer"
start_index=0
end_index=2
echo $start_index
echo $end_index
root_paths=("./data/tats")
data_paths=("Climate.csv") 
# root_paths=("./data/multimodal/Energy")
# data_paths=("US_GasolinePrice_Week.csv") 

seq_len=24
pred_lengths=(12 10 8 6)
seeds=(2022 2023 2024)
use_fullmodel=0
length=${#root_paths[@]}

mm_emb_size=4
mm_hidden_size=16
e_layers=2
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.0005
learning_rate2=0.0025
batch_size=32
train_epochs=50
patience=20
n_ts_features=1
text_emb=8
in_dim=1
prompt_weight=0.5
text_dropout=0.2
dropout=0.1
llm_emb_size=768
d_model=512
d_ff=1024
# llm_path=/home/s225250685/workspace/Huggingface/google-bert/bert-base-uncased
llm_path=/scratch/s225250685/Huggingface/openai-community/gpt2
# llm_path=/home/user11/hiepnh/Huggingface/google-bert/bert-base-uncased
echo $in_dim

for seed in "${seeds[@]}"
do
  for ((i=0; i<$length; i++))
  do
    for pred_len in "${pred_lengths[@]}"
    do
      root_path=${root_paths[$i]}
      data_path=${data_paths[$i]}
      model_id=$(basename ${data_path})

      echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${pipeline_name}_${model_name}_${model_id}_${seed}_${seq_len}_${pred_len}_${mm_emb_size}_${d_model}_${batch_size}_${learning_rate}_${prompt_weight}_${learning_rate2} \
        --model $model_name \
        --data custom \
        --train_epochs $train_epochs \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --batch_size $batch_size \
        --pred_len $pred_len \
        --des 'Exp' \
        --seed $seed \
        --type_tag "#F#" \
        --text_len 2 \
        --prompt_weight $prompt_weight \
        --learning_rate $learning_rate \
        --learning_rate2 $learning_rate2 \
        --pool_type "avg" \
        --save_name "result_health_bert" \
        --llm_model GPT2 \
        --huggingface_token 'NA'\
        --use_fullmodel $use_fullmodel \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in $in_dim \
        --dec_in $in_dim\
        --c_out $in_dim \
        --method 'SpecTF'\
        --n_ts_features $n_ts_features \
        --text_emb $text_emb \
        --patience $patience \
        --d_model $d_model \
        --d_ff $d_ff \
        --llm_path $llm_path \
        --mm_hidden_size $mm_hidden_size \
        --mm_emb_size $mm_emb_size \
        --features S \
        --proj_per_freq \
        --text_dropout $text_dropout \
        --dropout $dropout  \
        --fuse_history \
        --use_product \
        --llm_emb_size $llm_emb_size
    done
  done
done
