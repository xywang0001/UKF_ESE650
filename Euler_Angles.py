
# coding: utf-8

# In[9]:


import math
import numpy as np


# In[10]:


def euler_angles(q):

     r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]),1 - 2*(q[1]**2 + q[2]**2))
     p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
     y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1 - 2*(q[2]**2 + q[3]**2))
     return np.array([r, p, y])

