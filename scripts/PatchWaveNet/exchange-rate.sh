if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchWaveNet

root_path_name=./dataset/exchange_rate
data_path_name=exchange_rate.csv
model_id_name=exchange_rate
data_name=custom

random_seed=2021
pred_len=96
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 1\
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --d_conv 2\
      --patch_len 4\
      --stride 4 \
      --des 'Exp' \
      --train_epochs 10\
      --patience 2\
      --loss_flag 2\
      --itr 1 --batch_size 32 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

pred_len=192
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 4\
      --stride 4 \
      --d_conv 2 \
      --d_state 16 \
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --loss_flag 2\
      --itr 1 --batch_size 128 --learning_rate 0.0001

pred_len=336
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 4\
      --d_conv 2 \
      --stride 4 \
      --des 'Exp' \
      --d_state 16 \
      --train_epochs 10\
      --patience 3\
      --loss_flag 2\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 256 --learning_rate 0.0001

pred_len=720
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 4 \
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 4\
      --stride 4 \
      --d_conv 2 \
      --d_state 16 \
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --loss_flag 2\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 16 --learning_rate 0.0001