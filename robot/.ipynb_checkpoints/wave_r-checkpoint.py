# Choregraphe simplified export in Python.
IP='192.168.1.65'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names.append("RElbowRoll")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889, 3.33333])
keys.append([0.589921, 1.35236, 1.29784, 1.37763, 1.29784, 1.37763, 1.37751, 0.314159])

names.append("RElbowYaw")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889])
keys.append([1.29852, 1.61682, 1.95651, 1.309, 1.74882, 1.69646, 1.39285])

names.append("RHand")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889, 3.33333])
keys.append([0.456942, 0.756942, 0.856942, 0.96942, 0.956942, 0.956942, 0.560457, 0.45])

names.append("RShoulderPitch")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889, 3.33333])
keys.append([0.435651, 0.216291, 0.145728, 0.147262, 0.145728, 0.147262, 0.841249, 1.49051])

names.append("RShoulderRoll")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889, 3.33333])
keys.append([-0.0843689, -0.081301, -0.139592, -0.00872665, -0.139592, -0.0907571, -0.0874369, -0.0541052])

names.append("RWristYaw")
times.append([0.666667, 1.11111, 1.33333, 1.88889, 2.11111, 2.44444, 2.88889, 3.33333])
keys.append([-1.1612, -1.44652, -1.05382, -1.69656, -1.05382, -1.69656, -1.49254, 0.169297])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  motion.setExternalCollisionProtectionEnabled("Arms", False)

  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err

