for ntrain in 10000 5000 1500
do
echo $ntrain
python main_transfer_analytical_sim.py --ntrain_samples $ntrain -c configs/analytical_transfer.yaml
done