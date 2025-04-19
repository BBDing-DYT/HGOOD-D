python main.py -exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -lr 0.00005 -cluster_num 5,1 -batch_size 203 -num_epoch 800 -alpha 0.0 -HPC
#-exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -lr 0.0001 -batch_size 203 -num_epoch 800 -alpha 0.2 -HPC
#loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean() * 0.001 - loss_IB2.mean() * 0.001
#-exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -lr 0.0001 -cluster_num 8,3 -batch_size 128 -num_epoch 800 -alpha 0.0 -HPC
#-exp_type oodd -DS_pair ogbg-molesol+ogbg-molmuv -lr 0.00005 -cluster_num 5,1 -batch_size 203 -num_epoch 800 -alpha 0.0 -HPC
