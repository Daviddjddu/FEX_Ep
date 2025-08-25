import os
import time
import random

# This one might be harder to obtain the right form since it has the interaction term(blend with the square term)
# May change the lr to 0.001
# Replace x3 by 1-x0-x1-x2, expand and clean up the expression tree depth2 for interation term

for i in range(1):
    for dim in [3]:
        gpu = 1
        whichDim = 1
        os.system('python controller_covid_1.py --epoch 100 --bs 10  --greedy 0.1 --gpu '+str(gpu)+' --ckpt Dim'+str(whichDim)+' --tree depth2_rmu3 --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 200000 --left -1 --right 1 --domainbs 5000 --bdbs 1000')
        
       