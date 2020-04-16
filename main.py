
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from estimate_rot import *
import time
#from scipy import io
s= time.time()
roll,pitch,yaw = estimate_rot(1)
e = time.time()

print (s-e)

plt.plot(roll)



