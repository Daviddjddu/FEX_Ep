import os
import time
import random

# run it twice, will get pretty nice form
# Make sure to change the data path in controller_ode.py
# Use the depth2_sub for the dimension with nonlinear term
for i in range(1):
    for dim in [4]:
        gpu = 0
        whichDim = 1 # make sure to change this for different dimension of this model
        os.system('python controller_ode.py --epoch 100 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt Dim'+str(whichDim)+' --tree depth2_rmu3 --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 200000 --left -1 --right 1 --domainbs 5000 --bdbs 1000')
        
        