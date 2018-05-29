robot_ip="192.168.1.166"
port=9559

from naoqi import ALProxy
import glob
import json

class Pepper():
    def __init__(self,ip=robot_ip,port=port):
        self.motion=ALProxy("ALMotion",ip,port)
        self.actions=load_act()
        camProxy = ALProxy("ALVideoDevice", ip, port)
        resolution = 1    # VGA
        colorSpace = 0   # Y channel

        upper_cam = camProxy.subscribeCamera("Ucam",0, resolution, colorSpace, 5)
        depth = camProxy.subscribeCamera("Dcam",2, resolution, colorSpace, 5)

        basic_awareness = ALProxy("ALBasicAwareness",ip, port)

        basic_awareness.setStimulusDetectionEnabled("People",True)
        basic_awareness.setStimulusDetectionEnabled("Movement",True)
        basic_awareness.setStimulusDetectionEnabled("Sound",True)
        basic_awareness.setStimulusDetectionEnabled("Touch",True)

        basic_awareness.setParameter("LookStimulusSpeed",0.7)
        basic_awareness.setParameter("LookBackSpeed",0.5)
        basic_awareness.setEngagementMode("FullyEngaged")
        basic_awareness.setTrackingMode("Head")

        self.tracker = ALProxy("ALTracker", ip, port)
        targetName = "Face"
        faceWidth = 0.1
        self.tracker.registerTarget(targetName, faceWidth)
        self.tracker.track(targetName)


    def perform(self,name,angles,times,isAbsolute=True):
        self.motion.setExternalCollisionProtectionEnabled("Arms", False)
        self.motion.angleInterpolation(name,angles,times,isAbsolute)
        self.motion.setExternalCollisionProtectionEnabled("Arms", True)

    def perform_act(self,name):
        act = self.actions[name]
        names=list()
        keys=list()
        times=list()
        for i in list(act.keys()):
            names.append(i)
            keys.append(act[i]['keys'])
            times.append(act[i]['times'])
        names=list(map(str,names))
        self.perform(names,keys,times)

def load_act():
    acts=dict()
    fils = glob.glob('actions/*.json')
    for fil in fils:
        act_j = json.load(open(fil))
        acts[act_j['name']]=act_j['frames']
    return acts
if __name__ == '__main__':
    robot=Pepper()
    robot.perform_act('bow')
