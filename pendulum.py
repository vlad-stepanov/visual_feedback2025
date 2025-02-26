import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np

physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
# physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

dt = 1/240 # pybullet simulation step
q0 = 0.0   # starting position (radian)
qd = np.pi/2   # desired position (radian)
maxTime = 5
logTime = np.arange(0, 5, dt)
logPos = np.zeros(len(logTime))
logVel = np.zeros(len(logTime))
logCtrl = np.zeros(len(logTime))

# get rid of all the default damping forces
# p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
# p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# # go to the starting position
# p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
# for _ in range(1000):
#     p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=2, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

kp = 40
ki = 40
kd = 10
err_int = 0.0
err_prev = 0.0
for idx in range(len(logTime)):
    q = p.getJointState(boxId, 1)[0]
    dq = p.getJointState(boxId, 1)[1]

    logPos[idx] = q
    logVel[idx] = dq

    err = q - qd
    # velocity control
    # propotional regulator is enough
    vel = -kp * err
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=vel, controlMode=p.VELOCITY_CONTROL)

    q2 = p.getJointState(boxId, 2)[0]
    logPos[idx] = q2
    err2 = q2 - qd
    # velocity control
    # propotional regulator is enough
    vel2 = -kp * err2
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=2, targetVelocity=vel2, controlMode=p.VELOCITY_CONTROL)

    # torque control
    # proportional part is not enough!
    err_int += err * dt
    err_diff = (err - err_prev) / dt
    trq = -kp * err - ki * err_int - kd * err_diff
    err_prev = err
    # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=trq, controlMode=p.TORQUE_CONTROL)
    logCtrl[idx] = trq

    p.stepSimulation()
    # time.sleep(dt)
# p.disconnect()

xyzPos = p.getLinkState(boxId, 3)[0]
print(xyzPos)

plt.subplot(3,1,1)
plt.plot(logTime, logPos, label="pos")
plt.plot([logTime[0],logTime[-1]], [qd, qd],'r--', label="ref")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(logTime, logVel, label="vel")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(logTime, logCtrl, label="ctrl")
plt.grid(True)
plt.legend()

plt.show()
