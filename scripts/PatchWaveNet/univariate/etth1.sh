if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
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
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --d_model 64 \
      --dropout 0.2\
      --patch_len 4\
      --stride 2\
      --conv_channel 32 \
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 1024 --learning_rate 0.0001

pred_len=192
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --d_model 64 \
      --dropout 0.2\
      --patch_len 4\
      --stride 2\
      --conv_channel 32 \
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 1024 --learning_rate 0.0001
for pred_len in 336
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --patch_len 16\
      --stride 8\
      --conv_channel 32 \
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 1024 --learning_rate 0.0001 #>logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log
done
pred_len=720
python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --patch_len 4\
      --stride 2\
      --conv_channel 32 \
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 1024 --learning_rate 0.0001