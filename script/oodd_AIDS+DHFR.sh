python main.py -exp_type oodd -DS_pair AIDS+DHFR -num_epoch 400 -batch_size 200 -cluster_num 10,1 -alpha 0.2 -HPC
#loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01
#if epoch % 10 == 0 and epoch > 200: