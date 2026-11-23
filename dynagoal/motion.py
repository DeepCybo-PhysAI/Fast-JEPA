"""Task-side motion and event timing; no policy changes or future-policy inputs."""
import math
import numpy as np

TASKS=('pusht','cube','reacher','tworoom')
# Amplitudes/speeds chosen in environment units before observing policy outcomes.
LEVELS={'pusht':(0.,8.,16.),'cube':(0.,.02,.04),'reacher':(0.,.1,.2),'tworoom':(0.,3.,5.)}
PERIOD=4.0

def sinusoidal_offset(t, amplitude, direction, period=PERIOD):
    return amplitude*math.sin(2*math.pi*t/period)*np.asarray(direction)

def point_segment_distance(point, start, end):
    point,start,end=map(lambda x:np.asarray(x,dtype=float),(point,start,end))
    delta=end-start
    u=np.clip(np.dot(point-start,delta)/max(np.dot(delta,delta),1e-20),0.,1.)
    return float(np.linalg.norm(point-(start+u*delta)))

def swept_disk_collision(agent0,agent1,obstacle0,obstacle1,radius):
    # Relative linear motion during one native control interval.
    return point_segment_distance(np.zeros(2),np.asarray(agent0)-obstacle0,np.asarray(agent1)-obstacle1)<radius

def ready_tick(observation_tick, seconds, dt):
    if seconds<0 or dt<=0:raise ValueError('Invalid timing')
    return observation_tick+math.ceil(max(0.,seconds/dt-1e-12))
