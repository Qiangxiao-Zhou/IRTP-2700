import h5py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize data
X, Y_1, Y_2 = [], [], []
Z_1, Z_2, W = [], [], []
Flag = []
actual_lengths_X, actual_lengths_Y_1, actual_lengths_Y_2 = [], [], []


flag=0

with h5py.File('IR_trajectory_power_data.mat', 'r') as data:
    motions = data['motions']
    
    # Iterate through 27 sets of data
    for i in range(27):
        motion_key = f'motion_{i}'
        motion_group = motions[motion_key]
        
        # Each set of data contains 100 trajectories-power data
        for j in range(100):
            path_key = f'path_{j}'
            path_group = motion_group[path_key]
            
            # Trajectory data from sampling
            traj_data = path_group['IR_trajectory_data']
            joint_position = traj_data['joint_position'][()].T
            compute_velocity = traj_data['joint_velocity'][()].T
            compute_acceleration = traj_data['joint_acceleration'][()].T
            joint_time = traj_data['time'][()].T
            
            # Power data measured by the power meter
            power_data = path_group['power_meter_measurements']
            pm_power = power_data['power'][()].T
            pm_time = power_data['time'][()].T
            pm_energy = power_data['total_energy'][()].item()
            
            # Power data measured by an oscilloscope
            osc_data = path_group['oscilloscope_measurements']
            osc_power = osc_data['power'][()].T
            osc_time = osc_data['time'][()].T
            osc_energy = osc_data['total_energy'][()].item()
            
            # Marked load mass
            workload = path_group['workload'][()].item()
            
            # Record actual length
            actual_lengths_X.append(joint_time.shape[0])
            actual_lengths_Y_1.append(pm_time.shape[0])
            actual_lengths_Y_2.append(osc_time.shape[0])

     
            # Feature concatenation
            features_traj = np.hstack([
                joint_position,
                compute_velocity,
                compute_acceleration,
                joint_time
            ])
            
            features_pm = np.hstack([pm_power, pm_time])
            features_osc = np.hstack([osc_power, osc_time])

            # Numbering data makes it easier to find specific data in tensors.
            flag = i * 100 + j
            
            
            X.append(features_traj)
            Y_1.append(features_pm)
            Y_2.append(features_osc)
            Z_1.append(pm_energy)
            Z_2.append(osc_energy)
            W.append(workload)
            Flag.append(flag)


# Padding to align dimensions
maxlen_X = max(actual_lengths_X)
maxlen_Y1 = max(actual_lengths_Y_1)
maxlen_Y2 = max(actual_lengths_Y_2)

X_pad = pad_sequences(X, maxlen=maxlen_X, dtype='float32', padding='post', value=9999.0)
Y1_pad = pad_sequences(Y_1, maxlen=maxlen_Y1, dtype='float32', padding='post', value=9999.0)
Y2_pad = pad_sequences(Y_2, maxlen=maxlen_Y2, dtype='float32', padding='post', value=9999.0)


Z_1 = np.array(Z_1).reshape(-1, 1)
Z_2 = np.array(Z_2).reshape(-1, 1)
W = np.array(W).reshape(-1, 1)
Flag = np.array(Flag).reshape(-1, 1)


# View the shape of the processed data
print(f"X: {X_pad.shape}, Y1: {Y1_pad.shape}, Y2: {Y2_pad.shape}")
print(f"Z1: {Z_1.shape}, Z2: {Z_2.shape}, W: {W.shape}, Flag: {Flag.shape}")
print(f"len_X: {len(actual_lengths_X)}, len_Y1: {len(actual_lengths_Y_1)}, len_Y2: {len(actual_lengths_Y_2)}")


# Save data
np.savez('processed_IR_data.npz',
         X=X_pad, Y1=Y1_pad, Y2=Y2_pad,
         Z1=Z_1, Z2=Z_2, W=W, Flag=Flag,
         len_X=actual_lengths_X,
         len_Y1=actual_lengths_Y_1,
         len_Y2=actual_lengths_Y_2)
