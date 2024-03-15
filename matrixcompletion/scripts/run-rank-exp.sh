for rank1 in 2 5 10 15
do
echo $rank1
python main.py --rank1 $rank1 -c configs/variable-rank.yaml
done

#for ntrain in 1000 1500 2000 3000 4000
#do
#echo $ntrain
#python main.py --ntrain_samples $ntrain -c configs/variable-depth.yaml
#done
