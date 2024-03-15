for ntrain in 1000 1500 2000 3000 4000
do
echo $ntrain
python main.py --ntrain_samples $ntrain -c configs/variable-depth.yaml
done
