for depth in 1 2 3
do
echo $depth
python main-num-examples.py --depth $depth -c configs/rebuttal_iclr/num-examples-depth.yaml
done
