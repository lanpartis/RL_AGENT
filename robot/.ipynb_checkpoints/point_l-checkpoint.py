# Choregraphe simplified export in Python.
IP='192.168.1.65'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()


names.append("LElbowRoll")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([-0.490874, -0.76699, -0.409573, -0.409573, -0.76699, -0.490874])

names.append("LElbowYaw")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([-1.28087, -1.21798, -1.23025, -1.23025, -1.21798, -1.28087])

names.append("LHand")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([0.537786, 0.529877, 0.98, 0.98, 0.529877, 0.537786])

names.append("LShoulderPitch")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([1.63062, 0.972544, 0.558369, 0.558369, 0.972545, 1.63062])

names.append("LShoulderRoll")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([0.193282, 0.0628932, 0.0920389, 0.0920389, 0.0628931, 0.193281])

names.append("LWristYaw")
times.append([0.56, 0.96, 1.44, 2, 2.32, 2.64])
keys.append([-0.0858622, -0.72554, -1.80087, -1.80087, -0.72554, -0.0858622])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err
