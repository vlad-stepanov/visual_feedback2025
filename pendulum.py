import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np

# physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

numJoints = p.getNumJoints(boxId)
for idx in range(numJoints):
    print(f"{idx} {p.getJointInfo(boxId, idx)[1]} {p.getJointInfo(boxId, idx)[12]}")

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
qd = np.pi/2   # desired position (radian)
maxTime = 5
logTime = np.arange(0, 5, dt)
logPos = np.zeros(len(logTime))
logVel = np.zeros(len(logTime))
logCtrl = np.zeros(len(logTime))

logX = np.zeros(len(logTime))
logZ = np.zeros(len(logTime))

jointIndices = [1,2]

# get rid of all the default damping forces
# p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
# p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[q0,q0], controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=2, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

linkState = p.getLinkState(boxId, linkIndex=3)
x0 = linkState[0][0]
z0 = linkState[0][2]

xd = 0.1
zd = 1.0
L = 0.5

kp = 40
ki = 40
kd = 10
err_int = 0.0
err_prev = 0.0
for idx in range(len(logTime)):
    th1 = p.getJointState(boxId, 1)[0]
    dth1 = p.getJointState(boxId, 1)[1]

    logPos[idx] = th1
    logVel[idx] = dth1

    err = th1 - qd
    # ===== VELOCITY CONTROL =====
    # propotional regulator is enough
    vel1 = -kp * err

    th2 = p.getJointState(boxId, 2)[0]
    dth2 = p.getJointState(boxId, 2)[1]
    logPos[idx] = th2
    err2 = th2 - qd
    vel2 = -kp * err2
    # p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[vel1, vel2], controlMode=p.VELOCITY_CONTROL)

    # ===== CARTESIAN CONTROL =====
    # go to cartesian pose via inverse Jacobian
    linkState = p.getLinkState(boxId, linkIndex=3)
    xSim2 = linkState[0][0]
    zSim2 = linkState[0][2]

    logX[idx] = xSim2
    logZ[idx] = zSim2

    # Jacobian
    J = np.array([[-L*np.cos(th1)-L*np.cos(th1+th2), -L*np.cos(th1+th2)],
                  [L*np.sin(th1)+L*np.sin(th1+th2), L*np.sin(th1+th2)]])

    w = 10*np.linalg.inv(J) @ -np.array([[xSim2-xd],[zSim2-zd]])
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=[w[0,0],w[1,0]], controlMode=p.VELOCITY_CONTROL)


    # ===== TORQUE CONTROL =====
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

# plot joint coords
plt.figure()
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

# plot cartesian coords

plt.figure()
plt.subplot(2,1,1)
plt.plot(logTime, logX, label="X")
plt.plot([logTime[0],logTime[-1]], [xd, xd],'r--', label="ref")
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(logTime, logZ, label="Z")
plt.plot([logTime[0],logTime[-1]], [zd, zd],'r--', label="ref")
plt.grid(True)
plt.legend()

# plot XZ
plt.figure()
plt.plot(logX, logZ, label="XZ")
plt.plot([x0, xd], [z0, zd],'r--', label="ref")
plt.grid(True)
plt.legend()


plt.show()
