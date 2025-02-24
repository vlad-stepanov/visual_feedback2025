import pybullet as p
import time
import pybullet_data

dt = 1/240 # pybullet simulation step
q0 = 0.0   # starting position (radian)
qd = 1.5708   # desired position (radian)
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

# get rid of all the default damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# # go to the starting position
# p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
# for _ in range(1000):
#     p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

kp = 20
ki = 10
kd = 5
err_int = 0.0
err_prev = 0.0
while True:
    q = p.getJointState(boxId, 1)[0]
    err = q - qd
    # velocity control
    # propotional regulator is enough
    # vel = -kp * err
    # p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=vel, controlMode=p.VELOCITY_CONTROL)

    # torque control
    # proportional part is not enough!
    err_int += err * dt
    err_diff = (err - err_prev) / dt
    trq = -kp * err - ki * err_int - kd * err_diff
    err_prev = err
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=trq, controlMode=p.TORQUE_CONTROL)

    p.stepSimulation()
    time.sleep(dt)
p.disconnect()
