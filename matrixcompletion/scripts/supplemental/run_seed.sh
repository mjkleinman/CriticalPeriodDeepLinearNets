for seed in 2 3 4 # already ran 1
do
echo $seed
python main.py --seed $seed -c configs/variable-depth.yaml
done