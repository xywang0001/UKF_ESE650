#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

#from ukf_functions import compute_sigma_pts, process_model, prediction, measurement_model,update
#from Load_Data import load_imu
#import numpy as np
#from Euler_Angles import euler_angles

from scipy import io
import os
import numpy as np
import math

def estimate_rot(data_num=1):
    
    #from scipy import io
    #import os

    #Data process/ Loading_Data
    
    imu_vals, imu_ts = load_imu(data_num)
    
    # Unscented Kalman Filter
    # init
    qk = np.array([1,0,0,0]) # last mean in quaternion
    Pk = np.identity(3) * 0.1 # last cov in vector
    Q = np.identity(3) * 2 # process noise cov
    R = np.identity(3) * 2 # measurement noise cov

    time = imu_ts.shape[0]
    ukf_euler = np.zeros((time, 3))  # represent orientation in euler angles
    
    for t in range(time):
        # extract sensor data
        acc = imu_vals[t,:3]
        gyro = imu_vals[t,3:]
        
        X = compute_sigma_pts(qk, Pk, Q)
        
        if t == time-1: # last iter
            dt = np.mean(imu_ts[-10:] - imu_ts[-11:-1])
        else:
            dt = imu_ts[t+1] - imu_ts[t]
            
        Y = process_model(X, gyro, dt)

        q_pred, P_pred, W = prediction(Y, qk)
        
        
        vk, Pvv, Pxz = measurement_model(Y, acc, W, R)

        # Update
        K = np.dot(Pxz,np.linalg.inv(Pvv)) # Kalman gain
        qk, Pk = update(q_pred, P_pred, vk, Pvv, K)
        
        #print (qk.shape)
        
        ukf_euler[t, :] = euler_angles(qk[0])
        
    roll = ukf_euler[:,0]
    pitch = ukf_euler[:,1]
    yaw = ukf_euler[:,2]
    
    return roll,pitch,yaw

def load_imu(number):
    # Get the corresponding raw data
    filename = os.path.join(os.path.dirname(__file__),  "imu/imuRaw" + str(number) + ".mat")
    imuRaw = io.loadmat(filename)

    # Extract data
    imu_vals = np.array(imuRaw["vals"])
    imu_ts = np.array(imuRaw["ts"]).T
    
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

def vec2quat(vec):
    # exp mapping
    m = vec.shape[0]
    r = vec/2
    q1 = np.zeros((m,1))
    q1 = np.hstack([q1,r])
    q = quat_exp(q1)
    return q

def quat2vec(q):
    # log mapping
    r = (2*quat_log(q))[:,1:]
    return r

def quat_norm(q):
    m = q.shape[0]
    q_n = np.linalg.norm(q, axis = 1)
    q_n = np.transpose(q_n)
    q_n = q_n.reshape(m,1)
    return q_n

def quat_normalize(q):
    q_norm = np.copy(q)
    q_norm /= quat_norm(q_norm)
    return q_norm

def quat_conjugate(q):
    q_conj = np.copy(q) 
    q_conj[:,1:] *= -1
    return q_conj

def quat_inverse(q):
    q_conj = quat_conjugate(q)
    q_norm = quat_norm(q)
    return q_conj/(q_norm**2)

def quat_multiply(p,q):
    
    m = p.shape[0]
    
    p0 = p[:,0].reshape((m,1))
    pv = p[:,1:]
    q0 = q[:,0].reshape((m,1))
    qv = q[:,1:]

    r0 = np.multiply(p0, q0) - np.diagonal(np.dot(pv, np.transpose(qv))).reshape(m,1)
    rv = np.multiply(p0, qv) + np.multiply(q0, pv) + np.cross(pv, qv)


    return np.hstack([r0,rv])

def quat_exp(q):
    
    m = q.shape[0]
    
    #q = q.reshape((m,4))
    
    q0 = q[:,0].reshape(m,1)
    qv = q[:,1:]
    
    qvnorm = np.linalg.norm(qv, axis = 1)
    #qvnorm = np.transpose(qvnorm)
    qvnorm = qvnorm.reshape(m,1)

    z0 = np.multiply(np.exp(q0) , np.cos(qvnorm))
    
    #epsilon = 1e-16
    
    #sc = qvnorm + epsilon
    qv = np.divide(qv, qvnorm, out=np.zeros_like(qv), where=qvnorm!=0)
    q0 = np.exp(q0)
    #q0 = q0.reshape(m,1)
    
    zv = q0*qv*np.sin(qvnorm)
    return np.hstack([z0,zv])

def quat_log(q):
    
    m = q.shape[0]
    
    qnorm = quat_norm(q)
    
    q0 = q[:,0]
    q0 = q0.reshape(m,1)
    qv = q[:,1:]
    qvnorm = np.linalg.norm(qv, axis = 1)
    qvnorm = qvnorm.reshape(m,1)
    #qvnorm = np.transpose(qvnorm)
    
    z0 = np.log(qnorm)
    qv = qv.astype(float)
    qvnorm = qvnorm.astype(float)
    zv = np.divide(qv, qvnorm, out=np.zeros_like(qv), where=qvnorm!=0) *np.arccos(q0 / qnorm)
    
    return np.hstack([z0,zv])


def quat_avg(q_set, qt):
    
    qt = qt.reshape(1,4)
    n = q_set.shape[0]

    epsilon = 1E-3
    max_iter = 800

    
    
    
    for t in range(max_iter):
        
        qt_m = np.tile(qt, (n,1))
        q_err = quat_normalize(quat_multiply(q_set, quat_inverse(qt_m)))

        v_err = quat2vec(q_err)
        v_norm = np.linalg.norm(v_err, axis = 1)
        v_norm = v_norm.reshape(n,1)
        v_norm = v_norm.astype(float)

        temp = -np.pi + np.mod(v_norm + np.pi, 2 * np.pi)
        temp = temp.astype(float)

        err_vec = np.divide(temp, v_norm, out=np.zeros_like(temp), where=v_norm!=0) *v_err

        err = np.mean(err_vec, axis=0)
        err = err.reshape(1,3)
        qt = quat_normalize(quat_multiply(vec2quat(err), qt))
        

        if np.linalg.norm(err) < epsilon:
            break

    return qt, err_vec

def compute_sigma_pts(q, P, Q):
    
    n = P.shape[0]

    # compute distribution around zero, apply noise before process model
    S = np.linalg.cholesky(P + Q)
    Xpos = S * np.sqrt(2*n)
    Xneg = -S * np.sqrt(2*n)
    
    W = np.hstack((Xpos, Xneg))
    W = np.transpose(W)
    qw_m = vec2quat(W)
    q_m = np.tile(q, (2*n, 1))

    X = quat_multiply (q_m, qw_m)    
    # add mean, 2n+1 sigma points in total
    X = np.vstack((q, X))

    return X

def process_model(X, gyro, dt):
    n = X.shape[0]
    #Y = np.zeros((n,4))

    # compute delta quaternion
    gyro = (gyro * dt).reshape(1,3)
    qdelta = vec2quat(gyro)
    qd_m = np.tile(qdelta, (n,1))
    
    Y = quat_multiply(X, qd_m)
    
    return Y

def prediction(Y, qk):
    n = Y.shape[0]
    qk = qk.reshape(1,4)
    # compute mean (in quaternion)
    q_pred, W = quat_avg(Y, qk)

    # compute covariance (in vector)
    P_pred = np.zeros((3, 3))
    for i in range(n):
        P_pred += np.outer(W[i,:], W[i,:])
    P_pred /= n

    return q_pred, P_pred, W

def measurement_model(Y, acc, W, R):
    n = Y.shape[0]

    # define world gravity in quaternion
    g_q = np.array([0, 0, 0, 1])

    Z = np.zeros((n, 3))
    q_inv = quat_inverse(Y)
    g_q = np.tile(g_q, (n,1))
    Z = quat_multiply(quat_multiply(q_inv,g_q),Y)[:,1:]
    # measurement mean
    zk = np.mean(Z, axis=0)
    zk /= np.linalg.norm(zk)

    # measurement cov and correlation
    Pzz = np.zeros((3, 3))
    Pxz = np.zeros((3, 3))
    Z_err = Z - zk
    for i in range(n):
        Pzz += np.outer(Z_err[i,:], Z_err[i,:])
        Pxz += np.outer(W[i,:], Z_err[i,:])
    Pzz /= n
    Pxz /= n

    # innovation
    acc /= np.linalg.norm(acc)
    vk = acc - zk
    Pvv = Pzz + R

    return vk, Pvv, Pxz

def update(q_pred, P_pred, vk, Pvv, K):
    # note: q_pred, P_pred are in quaternion, while vk, Pvv in vector
    q_gain = vec2quat((K.dot(vk)).reshape(1,3))
    q_gain = q_gain.reshape(1,4)
    q_pred = q_pred.reshape(1,4)
    q_update = quat_multiply(q_gain,q_pred)
    P_update = P_pred - K.dot(Pvv).dot(K.T)
    return q_update, P_update

def euler_angles(q):

     r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]),1 - 2*(q[1]**2 + q[2]**2))
     p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
     y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1 - 2*(q[2]**2 + q[3]**2))
     return np.array([r, p, y])