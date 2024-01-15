echo "seed:         ${seed:=4}"
echo "input_window: ${input_window:=96}"
echo "output_window:${output_window:=336}"
echo "valid_ratio:  ${valid_ratio:=0.2}"
echo "test_ratio:   ${test_ratio:=0.2}"
echo "batch_size:   ${batch_size:=64}"
echo "device:       ${device:=0}"
echo "hidden_size:  ${hidden_size:=512}"
echo "num_layers:   ${num_layers:=2}"
echo "output_size:  ${output_size:=7}"
echo "lr:           ${lr:=1e-5}"
echo "epochs:       ${epochs:=1000}"
echo "patience:     ${patience:=10}"
echo "model_path:   ${model_path:=BiLSTM.long.v1.4}"
#echo "model_path:   ${model_path:=Transformer_new.long.v1.0}"
#echo "model_path:   ${model_path:=Transformer_new.short.v1.0}"
echo "model:        ${model:=LSTM}"

for SEED in 0 1 2 3 4
do
  python -u run_new.py --seed $SEED \
                   --input_window $input_window \
                   --output_window $output_window \
                   --valid_ratio $valid_ratio \
                   --test_ratio $test_ratio \
                   --batch_size $batch_size \
                   --device $device \
                   --hidden_size $hidden_size \
                   --num_layers $num_layers \
                   --output_size $output_size \
                   --lr $lr \
                   --epochs $epochs \
                   --patience $patience \
                   --model_path $model_path.seed$SEED \
                   --model $model
done
echo "Finish"