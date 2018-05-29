# Choregraphe simplified export in Python.
IP='192.168.1.65'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names.append("HeadPitch")
times.append([0.24, 0.76, 1.24, 1.52, 1.88, 2.32, 2.6])
keys.append([0.523088, 0.623083, 0.523088, 0.523087, 0.404916, 0.179769, 0.00174533])

names.append("HeadYaw")
times.append([0.24, 0.76, 1.24, 1.52, 1.88, 2.32, 2.6])
keys.append([0, 0, 0, 0, 0, 0, 0])

names.append("HipPitch")
times.append([0.24, 0.76, 1.24, 1.52, 1.88, 2.32, 2.6])
keys.append([-0.0383496, -0.429351, -0.724312, -0.724312, -0.429351, -0.010472, -0.010472])

names.append("HipRoll")
times.append([0.24, 0.76, 1.24, 1.52, 1.88, 2.32, 2.6])
keys.append([-0.0184078, -0.0184078, -0.0184078, -0.0184078, -0.0184078, -0.0184078, -0.0184078])

names.append("KneePitch")
times.append([0.24, 0.76, 1.24, 1.52, 1.88, 2.32, 2.6])
keys.append([-0.0122719, -0.0122719, -0.0122719, -0.0122719, -0.0122719, -0.0122719, -0.0122719])

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err



