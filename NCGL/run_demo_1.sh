for d in  'Reddit-CL'
do
  for mt in 'gem' 'joint' 'ewc' 'lwf'  'mas' 'twp' 'bare' 'ergnn'
  do
    for IL in 'taskIL'
    do
    for inter in 'False'
    do
python train.py --dataset $d \
       --method $mt \
       --backbone GCN \
       --gpu 1 \
       --ILmode $IL \
       --inter-task-edges $inter \
       --epochs 200 \
       --minibatch True \
       --batch_size 2000 \
       --sample_nbs True \
       --n_nbs_sample 10,25 \
       --repeats 5 \
       --ratio_valid_test 0.4 0.4
  done
done
done
done