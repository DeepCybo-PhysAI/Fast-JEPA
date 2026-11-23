import numpy as np
from . import dynamics as core
from .motion import sinusoidal_offset
PERIOD=20.
def slow_motion(t,amplitude,direction,period=PERIOD):
    return sinusoidal_offset(t,amplitude,direction,period)

class RefinedRunner(core.DynamicRunner):
    def initialize_dynamic(self,case,level):
        self._door_base=None
        return super().initialize_dynamic(case,level)

    def update_goal(self,t):
        if self.task=='tworoom':
            before=self.physical_state().copy()
            if self._door_base is None:
                self._door_base=np.asarray(self.raw.door_positions).copy()
                self._door_index=int(np.argmin(np.abs(self._door_base[:self.raw.num_doors]-self.door_center)))
            sizes=np.asarray(self.raw.door_sizes)
            lo=self.raw.BORDER_SIZE+sizes[self._door_index]+8
            hi=self.raw.IMG_SIZE-self.raw.BORDER_SIZE-sizes[self._door_index]-8
            self.raw.door_positions[self._door_index]=float(np.clip(self._door_base[self._door_index]+self._door_amplitude*np.sin(2*np.pi*t/PERIOD),lo,hi))
            image=self.raw._render_frame(agent_pos=self.raw.target_position).cpu().numpy().transpose(1,2,0)
            self.goal={'goal':core.resize(image)[None,None]};self.world.infos.update(self.goal)
            assert np.array_equal(before,self.physical_state())
            return
        if self.task=='cube' and self.level:
            before=self.physical_state().copy()
            pos=self.base_goal_position.copy();pos[:2]+=slow_motion(t,self.level,self.case['direction'])
            bounds=self.raw.variation_space['cube']['goal_position']
            pos[:2]=np.clip(pos[:2],np.asarray(bounds.low)[0],np.asarray(bounds.high)[0])
            self.raw.set_target_pos(0,pos,self.base_goal_quat)
            q=self.goal_qpos.copy();q[self.object_address:self.object_address+3]=pos
            image=self.goal_scene(q,self.goal_qvel)
            self.current_target=pos;self.goal={'goal':core.resize(image)[None,None]};self.world.infos.update(self.goal)
            assert np.array_equal(before,self.physical_state())
            return
        return super().update_goal(t)

    def dynamic_episode(self,case,level,latency=True,steps=100,feedback=25):
        if self.task=='tworoom':
            self._door_amplitude=level
            record=super().dynamic_episode(case,0,latency,steps,feedback)
            record['level']=level;record['profile']='sliding_door'
            return record
        record=super().dynamic_episode(case,level,latency,steps,feedback)
        record['profile']='slow_motion'
        return record
