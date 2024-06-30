num_heads="0 1 2 3"
num_layers="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"

for num_head in $num_heads
do
    for num_layer in $num_layers
    do
        export CUDA_VISIBLE_DEVICES=0
        export HEAD_NUM=$num_head
        export LAYER_NUM=$num_layer
        export CALIBRATION=1
        export BETA=0.4
        export THRES=0.1
        echo "Testing Head Num: $HEAD_NUM, Testing Layer Num: $LAYER_NUM"
        python main.py \
               --ckpt_dir "meta-llama/Llama-2-7b-chat-hf" \
               --model_type "llama" \
               --calibrate 1 \
               --do_augmentation 1 \
               --num_samples 900 \
               --output_dir './calibration_results/cal.log'
    done
done
