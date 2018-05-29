# Choregraphe simplified export in Python.
IP='192.168.1.166'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names.append("RElbowRoll")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.12, 2.36, 2.56, 2.76, 3.6])
keys.append([0.518486, 0.527689, 0.846757, 1.09607, 0.786932, 0.912807, 0.63879, 0.932006, 0.994838, 0.923279, 0.228638])

names.append("RElbowYaw")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.12, 2.36, 2.56, 2.76, 3.6])
keys.append([1.21184, 1.14895, -0.0997088, -0.966408, -0.966408, -0.951068, -1.01055, -0.951068, -0.951068, -0.951068, 1.29852])

names.append("RHand")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.36, 2.56, 2.76, 3.6])
keys.append([0.388401, 0.565026, 0.782074, 0.9, 0.9, 0.98, 0.94, 0.77, 0.86, 0.565026])

names.append("RShoulderPitch")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.12, 2.36, 2.56, 2.76, 3.6])
keys.append([1.56926, 1.24252, -0.254641, -1.07379, -1.07379, -0.958186, -1.11527, -0.853466, -0.825541, -1.08452, 1.48178])

names.append("RShoulderRoll")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.36, 2.56, 2.76, 3.6])
keys.append([-0.15033, -0.18675, -0.213223, -0.222427, -0.222427, -0.18101, -0.18101, -0.18101, -0.18101, -0.205949])

names.append("RWristYaw")
times.append([0.4, 0.6, 1.04, 1.44, 1.6, 1.88, 2.36, 2.56, 2.76, 3.6])
keys.append([0.116542, -0.898966, 0.207048, 1.10751, 1.10751, 0.954106, 0.954107, 0.954107, 0.954107, 0.125664])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  motion.setExternalCollisionProtectionEnabled("Arms", False)
  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err


