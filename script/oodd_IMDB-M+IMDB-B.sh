python main.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -lr 0.0001 -batch_size 150 -lr 0.0001 -num_epoch 200 -cluster_num 5,1 -alpha 0.8 -HPC
#loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01
#x_em = torch.cat([data.x, data.x_s], dim=1)