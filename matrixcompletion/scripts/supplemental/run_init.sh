for init_scale in 0.1 0.01 0.001 0.0001 #0.001 already ran
do
echo $init_scale
python main.py --init_scale $init_scale -c configs/init.yaml
done