for depth in 1 2 3
do
echo $depth
python main.py --depth $depth -c configs/variable-depth.yaml
done
