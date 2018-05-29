import rospy
import numpy as np
import Utils.image as img_util
import time
from sensor_msgs.msg import Image
fishcam='/camera/fisheye/image_raw'
rgbcam='/camera/color/image_raw'
depthcam='/camera/depth/image_raw'
time_len=3
FPS=8
class State():
    def __init__(self,maxlen=24):
        self.state=list()
        self.size=24
    def put(self,img):
        self.state.append(img)
        if len(self.state)>self.size:
            del self.state[0]
    def show(self):
        if len(self.state)<self.size:
            return False,len(self.state)
        return np.array(self.state)

glob_states=dict()
glob_states[fishcam]=State()
glob_states[rgbcam]=State()
glob_states[depthcam]=State()
class Cam():
    def __init__(self,FPS=FPS,state=State(time_len*FPS)):
        self.FPS=FPS
        rospy.set_param('/camera/driver/color_width',480)
        rospy.set_param('/camera/driver/depth_width',480)
        rospy.set_param('/camera/driver/color_height',480)
        rospy.set_param('/camera/driver/depth_height',480)
        
        rospy.init_node('cam_listener',anonymous=True)
        rospy.Subscriber(fishcam,Image,callback_fish)
        rospy.Subscriber(rgbcam,Image,callback_rgb)
        rospy.Subscriber(depthcam,Image,callback_depth)
        rospy.set_param(fishcam+'/theora/keyframe_frequency',FPS)
        rospy.set_param(rgbcam+'/theora/keyframe_frequency',FPS)
        rospy.set_param(depthcam+'/theora/keyframe_frequency',FPS)
    
    def get_size(self,name):
        return len(glob_states[name].state)
    def get_state(self,name,delay):
        time.sleep(delay)
        return glob_states[name].show()

def callback_fish(data):
    data = img_util.image_to_numpy(data)
    glob_states[fishcam].put(data)
def callback_rgb(data):
    data = img_util.image_to_numpy(data)
    glob_states[rgbcam].put(data)
def callback_depth(data):
    data = img_util.image_to_numpy(data)
    glob_states[depthcam].put(data)
if __name__ == '__main__':
    cam = Cam()
    print(cam.get_state(fishcam,3).shape)