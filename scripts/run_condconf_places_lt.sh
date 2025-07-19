for SEED in 1 2 3 4 5 6 7 8 9 10
do
python run_condconf.py \
    --dataset_name places_lt \
    --model_name resnet152_places_lt \
    --seed $SEED \
    --temp_scaling \
    --score_fn aps
done