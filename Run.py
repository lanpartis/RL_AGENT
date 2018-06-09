from Q_agent import DQNAgent
from robot import Cam
from robot import Pepper
import thread
import time
ip='192.168.1.166'
robot = Pepper.Pepper(ip=ip)
cam = Cam.Cam()
Q_Agent = DQNAgent()
ep_file='RL_DATA/EP.data'
time_unit = 3
camera = Cam.fishcam
actions=robot.action_list
ep = int(open(ep_file).read())
time_stamp = 0
robot.perform_act('wait')
state = cam.get_state(camera,time_unit)

while True:
    s = time.time()
    act_num = Q_Agent.get_action_eps(state)
    time.sleep(1)
    thread.start_new_thread(robot.perform_act,(actions[act_num],))
    n_state = cam.get_state(camera,time_unit)
    Q_Agent.memorize(state,act_num,n_state,ep,time_stamp)
    state = n_state
    time_stamp+=1
