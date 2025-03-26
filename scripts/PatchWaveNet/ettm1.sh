if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchWaveNet

root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

for pred_len in 96 192
do
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
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.3\
      --itr 1 --batch_size 1024 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log
done

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
      --e_layers 1 \
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 1024 --learning_rate 0.0001

for pred_len in 720
do
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
      --d_conv 2 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 512 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log
done