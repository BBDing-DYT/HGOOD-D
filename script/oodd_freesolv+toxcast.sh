python main.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -lr 0.001 -batch_size 150 -num_epoch 800 -alpha 0.6 -HPC
#loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean() - loss_IB2.mean() * 0.001
#x_em = torch.cat([data.x, data.x_s], dim=1)