source /home/wzhangbu/anaconda3/etc/profile.d/conda.sh
conda activate timsseries
cd ~/Time-Series-Library


export CUDA_VISIBLE_DEVICES=3

model_name=Reformer # TimeMixer Autoformer

e_layers=2
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16


#python -u run.py \
#  --task_name anomaly_detection \
#  --is_training 1 \
#  --root_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data \
#  --data_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data \
#  --model_id m4_Monthly \
#  --model TimesNet \
#  --data Elderly_AD \
#  --features M \
#  --seq_len 200 \
#  --pred_len 0 \
#  --d_model 8 \
#  --d_ff 16 \
#  --e_layers 1 \
#  --enc_in 46 \
#  --c_out 46 \
#  --top_k 3 \
#  --anomaly_ratio 1 \
#  --batch_size 128 \
#  --train_epochs 40

#python -u run.py \
#  --task_name 'anomaly_detection' \
#  --is_training 1 \
#  --root_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data \
#  --seasonal_patterns 'Minutely' \
#  --model_id m4_Monthly \
#  --model $model_name \
#  --data Elderly \
#  --features M \
#  --e_layers $e_layers \
#  --pred_len 10 \
#  --label_len 10 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 46 \
#  --dec_in 8 \
#  --c_out 46 \
#  --batch_size 512 \
#  --d_model $d_model \
#  --d_ff 32 \
#  --des 'Exp' \
#  --itr 1 \
#  --lradj 'type3' \
#  --channel_independence 0 \
#  --seq_len 200 \
#  --learning_rate $learning_rate \
#  --train_epochs 200 \
#  --patience 40 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE' \
#  --data_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data \

#python -u run.py \
#  --task_name imputation \
#  --is_training 1 \
#  --root_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data \
#  --seasonal_patterns 'Minutely' \
#  --model_id m4_Monthly \
#  --model $model_name \
#  --data Elderly \
#  --features M \
#  --e_layers $e_layers \
#  --pred_len 10 \
#  --label_len 10 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 46 \
#  --dec_in 8 \
#  --c_out 46 \
#  --batch_size 512 \
#  --d_model $d_model \
#  --d_ff 32 \
#  --des 'Exp' \
#  --itr 1 \
#  --lradj 'type3' \
#  --channel_independence 0 \
#  --seq_len 200 \
#  --learning_rate $learning_rate \
#  --train_epochs 200 \
#  --patience 40 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE' \
#  --data_path /nfs/dataset/pathology/elderlycare/results_v7/cleaned_data
#  --data_path recording_2019_06_22_9_20_am/p_11.csv


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path /nfs/dataset/pathology/elderlycare/results_v7/DATASET \
  --seasonal_patterns 'Minutely' \
  --model_id m4_Monthly \
  --model $model_name \
  --data Elderly \
  --features M \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 46 \
  --dec_in 46 \
  --c_out 46 \
  --batch_size 256 \
  --d_model $d_model \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --seq_len 200 \
  --learning_rate $learning_rate \
  --train_epochs 1 \
  --patience 20 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --loss 'SMAPE' \
  --pred_len 5
  # --dec_in 8 or 46 \

###################################################################


#python -u run.py \
#  --task_name short_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/m4 \
#  --seasonal_patterns 'Yearly' \
#  --model_id m4_Yearly \
#  --model $model_name \
#  --data m4 \
#  --features M \
#  --e_layers $e_layers \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 128 \
#  --d_model $d_model \
#  --d_ff 32 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $learning_rate \
#  --train_epochs 50 \
#  --patience 20 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE'
#
#python -u run.py \
#  --task_name short_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/m4 \
#  --seasonal_patterns 'Quarterly' \
#  --model_id m4_Quarterly \
#  --model $model_name \
#  --data m4 \
#  --features M \
#  --e_layers $e_layers \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 128 \
#  --d_model $d_model \
#  --d_ff 64 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $learning_rate \
#  --train_epochs 50 \
#  --patience 20 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE'
#
#python -u run.py \
#  --task_name short_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/m4 \
#  --seasonal_patterns 'Daily' \
#  --model_id m4_Daily \
#  --model $model_name \
#  --data m4 \
#  --features M \
#  --e_layers $e_layers \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 128 \
#  --d_model $d_model \
#  --d_ff 16 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $learning_rate \
#  --train_epochs 50 \
#  --patience 20 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE'
#
#python -u run.py \
#  --task_name short_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/m4 \
#  --seasonal_patterns 'Weekly' \
#  --model_id m4_Weekly \
#  --model $model_name \
#  --data m4 \
#  --features M \
#  --e_layers $e_layers \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 128 \
#  --d_model $d_model \
#  --d_ff 32 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $learning_rate \
#  --train_epochs 50 \
#  --patience 20 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE'
#
#python -u run.py \
#  --task_name short_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/m4 \
#  --seasonal_patterns 'Hourly' \
#  --model_id m4_Hourly \
#  --model $model_name \
#  --data m4 \
#  --features M \
#  --e_layers $e_layers \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 128 \
#  --d_model $d_model \
#  --d_ff 32 \
#  --des 'Exp' \
#  --itr 1 \
#  --learning_rate $learning_rate \
#  --train_epochs 50 \
#  --patience 20 \
#  --down_sampling_layers $down_sampling_layers \
#  --down_sampling_method avg \
#  --down_sampling_window $down_sampling_window \
#  --loss 'SMAPE'