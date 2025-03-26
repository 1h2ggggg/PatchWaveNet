if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchWaveNet

root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather
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
      --enc_in 21 \
      --e_layers 1\
      --d_model 128\
      --patch_len 4\
      --stride 2\
      --des 'Exp' \
      --d_conv 2 \
      --train_epochs 100\
      --patience 10\
      --loss_flag 2\
      --itr 1 --batch_size 512 --learning_rate 0.001

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
      --enc_in 21 \
      --e_layers 1\
      --d_model 128\
      --patch_len 4\
      --stride 2\
      --des 'Exp' \
      --d_conv 2 \
      --train_epochs 200\
      --patience 20\
      --loss_flag 2\
      --itr 1 --batch_size 512 --learning_rate 0.001

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
      --enc_in 21 \
      --e_layers 1\
      --d_model 256\
      --patch_len 4\
      --stride 2\
      --des 'Exp' \
      --d_conv 2 \
      --train_epochs 100\
      --patience 20\
      --loss_flag 2\
      --itr 1 --batch_size 512 --learning_rate 0.001

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
      --enc_in 21 \
      --e_layers 1\
      --d_model 256\
      --patch_len 4\
      --stride 2\
      --des 'Exp' \
      --d_conv 2 \
      --train_epochs 100\
      --patience 20\
      --loss_flag 2\
      --itr 1 --batch_size 512 --learning_rate 0.001