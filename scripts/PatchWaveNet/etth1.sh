if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchWaveNet

root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021


pred_len=96
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --d_conv 2\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

pred_len=192
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --d_conv 2 \
      --conv_channel 4 \
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001

pred_len=336
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --d_conv 2\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001

pred_len=720
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 128 \
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --d_conv 2 \
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 1024 --learning_rate 0.0001

