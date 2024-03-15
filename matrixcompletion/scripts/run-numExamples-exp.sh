for nexample in 1250 1500 1750 2000
do
echo $nexample
python main-num-examples.py --ntrain_samples2 $nexample -c configs/num-examples.yaml
done
