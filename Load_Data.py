
# coding: utf-8

# In[18]:


from scipy import io
import os
import numpy as np


# In[20]:


def load_imu(number):
    # Get the corresponding raw data
    filename = os.path.join(os.path.dirname(__file__),  "imu/imuRaw" + str(number) + ".mat")
    imuRaw = io.loadmat(filename)

    # Extract data
    imu_vals = np.array(imuRaw["vals"])
    imu_ts = np.array(imuRaw["ts"]).T

    # Remove the biases
    #biases = np.mean(imu_vals[:, 1:300], axis=1, keepdims=True)
    #imu_val = imu_vals - biases
    # Extract acceleration
    #acc_data = data[0:3, :-chop] + np.array([[0], [0], [93]])
    # Extract angular velocity
    
    acc_x = -imu_vals[0,:]
    acc_y = -imu_vals[1,:]
    acc_z = imu_vals[2,:]
    acc = np.array([acc_x, acc_y, acc_z]).T

    Vref = 3300
    acc_sensitivity = 330
    acc_scale_factor = Vref/1023/acc_sensitivity
    acc_bias = np.mean(acc[:10], axis = 0) - (np.array([0,0,1])/acc_scale_factor)
    acc = (acc-acc_bias)*acc_scale_factor
    
    gyro_x = imu_vals[4,:]
    gyro_y = imu_vals[5,:]
    gyro_z = imu_vals[3,:]
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro_bias = np.mean(gyro[:10], axis = 0)
    gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)
    
    imu_vals = np.hstack((acc,gyro))

    return imu_vals,imu_ts

#def load_vicon(number):
#    filename = os.path.join(os.path.dirname(__file__),  "vicon/viconRot" + str(number) + ".mat")
#    vicon = io.loadmat(filename)
#    vicon_rot = vicon["rots"]
    
    
#    return rot
    