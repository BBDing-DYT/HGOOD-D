python main.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -lr 0.00001 -num_epoch 200 -batch_size 60 -alpha 0.2 -cluster_num 20,5 -HPC

#loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01