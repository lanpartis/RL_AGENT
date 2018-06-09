# Choregraphe simplified export in Python.
IP='192.168.1.166'
# Choregraphe simplified export in Python.
from naoqi import ALProxy
names = list()
times = list()
keys = list()

names =['HeadYaw','HeadPitch']
times=[[0.5,3.0],[0.5,3.0]]
keys = [[0.0,0.0],[-0.16,-0.16]]

try:
  # uncomment the following line and modify the IP if you use this script outside Choregraphe.
  motion = ALProxy("ALMotion", IP, 9559)
  motion.setExternalCollisionProtectionEnabled("Arms", False)

  # motion = ALProxy("ALMotion")
  motion.angleInterpolation(names, keys, times, True)
except BaseException, err:
  print err

