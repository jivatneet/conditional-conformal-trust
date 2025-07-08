for SEED in 1 2 3 4 5 6 7 8 9 10
do
python run_condconf.py \
    --dataset_name fitzpatrick17k \
    --model_name resnet18_fitzpatrick17k \
    --seed $SEED \
    --score_fn aps
done

