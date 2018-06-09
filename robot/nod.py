# Choregraphe simplified export in Python.
IP='192.168.1.166'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names.append("HeadPitch")
times.append([0.48, 0.64, 0.8, 0.88, 1.08, 1.24, 1.56])
keys.append([-0.174533, -0.221657, -0.146608, 0.0715585, 0.34383, 0.240855, -0.174533])

names.append("HeadYaw")
times.append([0.48, 0.64, 0.8, 1.08, 1.24, 1.56])
keys.append([0, 0, 0, 0, 0, 0])

names.append("HipPitch")
times.append([0.48, 0.64, 0.8, 1.08, 1.24, 1.56])
keys.append([-0.0383496, -0.0383496, -0.0383496, -0.0383496, -0.0383496, -0.0383496])

names.append("HipRoll")
times.append([0.48, 0.64, 0.8, 1.08, 1.24, 1.56])
keys.append([-0.0184078, -0.0184078, -0.0184078, -0.0184078, -0.0184078, -0.0184078])

names.append("KneePitch")
times.append([0.48, 0.64, 0.8, 1.08, 1.24, 1.56])
keys.append([-0.0122719, -0.0122719, -0.0122719, -0.0122719, -0.0122719, -0.0122719])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err
