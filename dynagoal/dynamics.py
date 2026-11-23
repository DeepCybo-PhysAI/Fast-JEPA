from pathlib import Path
import copy, hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from .deployment import Runner, _extract_init_goal
from .motion import sinusoidal_offset, swept_disk_collision, ready_tick, PERIOD

def resize(image):
    return np.array(Image.fromarray(np.asarray(image,dtype=np.uint8)).resize((224,224),Image.Resampling.BILINEAR),copy=True)

class DynamicRunner(Runner):
    def goal_scene(self,qpos,qvel):
        """Render a desired goal scene; restore full live integrator state exactly."""
        before=self.physical_state().copy();tm=self.physical_time()
        if self.task=='reacher':
            physics=self.raw.dmc_env.physics
            state=physics.get_state().copy()
            extra={k:np.asarray(getattr(physics.data,k)).copy() for k in ('ctrl','qacc_warmstart','qfrc_applied','xfrc_applied')}
            self.raw.set_state(np.asarray(qpos),np.asarray(qvel))
            image=self.raw.render()
            physics.set_state(state);physics.forward()
            for k,v in extra.items():getattr(physics.data,k)[:]=v
        else:
            import mujoco
            data=self.raw._data
            saved={k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied')}
            data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(self.raw._model,data)
            image=self.raw.render()
            for k,v in saved.items():getattr(data,k)[:]=v
            mujoco.mj_forward(self.raw._model,data)
            # mj_forward can touch solver warmstart; restore it after forward.
            data.qacc_warmstart[:]=saved['qacc_warmstart']
        assert np.array_equal(before,self.physical_state()),'Goal rendering mutated live state'
        assert tm==self.physical_time(),'Goal rendering changed clock'
        return resize(image)

    def obstacle(self,t):
        axis=self.wall_coordinate
        c=np.zeros(2);c[axis]=self.raw.wall_pos;c[1-axis]=self.door_center
        c[axis]+=28*np.sin(2*np.pi*t/PERIOD+self.case['phase'])
        return c

    def overlay(self,image,t):
        if self.task!='tworoom' or not self.level:return image
        im=Image.fromarray(np.asarray(image).copy());draw=ImageDraw.Draw(im)
        x,y=self.obstacle(t);r=self.level;draw.ellipse((x-r,y-r,x+r,y+r),fill=(0,0,0))
        return np.array(im,copy=True)

    def update_goal(self,t):
        before=self.physical_state().copy()
        if self.task=='cube':
            pos=self.base_goal_position.copy();pos[:2]+=sinusoidal_offset(t,self.level,self.case['direction'])
            self.raw.set_target_pos(0,pos,self.base_goal_quat)
            q=self.goal_qpos.copy();q[self.object_address:self.object_address+3]=pos
            image=self.goal_scene(q,self.goal_qvel) if self.level else self.static_goal_image
            self.current_target=pos
        elif self.task=='reacher':
            target=self.base_goal_position+sinusoidal_offset(t,self.level,self.case['direction'])
            # Keep the original unwrapped qpos topology; no angle wrapping repair.
            self.raw.set_target_qpos(target)
            image=self.goal_scene(target,np.zeros_like(target)) if self.level else self.static_goal_image
            self.current_target=target
        else:image=self.overlay(self.static_goal_image,t)
        assert np.array_equal(before,self.physical_state()),'Target update modified physical state'
        self.goal={'goal':np.array(image,copy=True)[None,None]}
        self.world.infos.update(self.goal)

    def observe(self,t):
        # Ground truth is used only to render current/desired RGB and score tasks.
        self.update_goal(t)
        self.world.infos['pixels']=resize(self.overlay(self.raw.render(),t))[None,None]

    def initialize_dynamic(self,case,level):
        self.case=case;self.level=float(level);super().initialize(case)
        self.static_goal_image=np.asarray(self.goal['goal'][0,0]).copy()
        _,goal,_=_extract_init_goal(self.dataset,[case['episode']],[case['start']],25)
        if self.task=='cube':
            self.base_goal_position=np.asarray(goal['goal_privileged_block_0_pos'][0]).copy()
            self.base_goal_quat=np.asarray(goal['goal_privileged_block_0_quat'][0]).copy()
            self.goal_qpos=np.asarray(goal['goal_qpos'][0]).copy();self.goal_qvel=np.asarray(goal['goal_qvel'][0]).copy()
            self.object_address=int(self.raw._model.joint('object_joint_0').qposadr[0])
        elif self.task=='reacher':self.base_goal_position=np.asarray(goal['goal_qpos'][0]).copy()
        elif self.task=='tworoom':
            target=np.asarray(self.raw.target_position);start=np.asarray(self.raw.agent_position)
            assert self.raw.wall_axis==0 or self.raw.wall_axis==1
            self.wall_coordinate=0 if self.raw.wall_axis==1 else 1
            assert (target[self.wall_coordinate]-self.raw.wall_pos)*(start[self.wall_coordinate]-self.raw.wall_pos)<0,'Need an actual door crossing'
            mids=.5*(target+start);self.door_center=float(min(self.raw.door_positions[:self.raw.num_doors],key=lambda p:abs(float(p)-mids[1-self.wall_coordinate])))
        self.observe(0.)

    def error_success(self):
        if self.task=='pusht':
            state=self.raw._get_obs();success,_=self.raw.eval_state(self.raw.goal_state,state)
            angle=abs(float(state[4]-self.raw.goal_state[4]))%(2*np.pi);angle=min(angle,2*np.pi-angle)
            err=max(np.linalg.norm(state[:4]-self.raw.goal_state[:4])/20,angle/(np.pi/9))
        elif self.task=='cube':err=float(np.linalg.norm(self.raw._data.joint('object_joint_0').qpos[:3]-self.current_target)/.04);success=err<=1
        elif self.task=='reacher':err=float(np.max(np.abs(self.raw.dmc_env.physics.data.qpos-self.current_target))/.05);success=err<1
        else:err=float(np.linalg.norm(np.asarray(self.raw.agent_position)-np.asarray(self.raw.target_position))/16);success=err<1
        return float(err),bool(success)

    def dynamic_episode(self,case,level,latency=True,steps=100,feedback=25):
        if self.task=="cube":self.raw._terminate_at_goal=False
        self.initialize_dynamic(case,level)
        pending=None;queue=[];plans=[];trajectory=[];collision=False;hit=None;waiting=0
        original_term=None
        if self.task=='reacher':
            original_term=self.raw.dmc_env.task.get_termination
            self.raw.dmc_env.task.get_termination=lambda physics:None
        start_time=self.physical_time()
        try:
            for tick in range(steps):
                t=tick*self.dt
                if pending is not None and tick>=pending[0]:queue=list(pending[1]);pending=None
                if tick%feedback==0:
                    assert pending is None,'Planning exceeded observation period; define overload explicitly'
                    self.observe(t)
                    actions,elapsed=self.make_plan(self.world.infos,self.history)
                    ready=ready_tick(tick,elapsed if latency else 0.,self.dt)
                    plans.append(dict(observation_step=tick,measured_seconds=elapsed,ready_step=ready,delay_steps=ready-tick))
                    if ready==tick:queue=list(actions)
                    else:pending=(ready,actions)
                if queue:action=queue.pop(0);source='plan'
                else:action=self.last_action.copy();source='hold';waiting+=1
                old_agent=np.asarray(self.raw.agent_position).copy() if self.task=='tworoom' else None
                self.step(action)
                if self.task=='pusht' and self.level:
                    # Kinematic conveyor advection, not an unmodelled extra control action.
                    pos=np.asarray(self.raw.block.position)+self.dt*self.level*np.asarray(case['direction'])
                    pos=np.clip(pos,40,self.raw.window_size-40)
                    self.raw.block.position=tuple(pos);self.raw.space.reindex_shapes_for_body(self.raw.block)
                nt=(tick+1)*self.dt
                self.update_goal(nt)
                if self.task=='tworoom' and self.level:
                    radius=float(self.raw.variation_space['agent']['radius'].value.item())+self.level
                    collision=swept_disk_collision(old_agent,np.asarray(self.raw.agent_position),self.obstacle(t),self.obstacle(nt),radius)
                err,success=self.error_success();success=success and not collision
                if success and hit is None:hit=tick+1
                trajectory.append(dict(step=tick+1,error_in_native_tolerances=err,success=success,collision=bool(collision),action=np.asarray(action).tolist(),source=source))
                if collision:break
            end_time=self.physical_time()
            if start_time is not None:assert end_time>start_time,'Physics auto-reset/stall'
        finally:
            if original_term is not None:self.raw.dmc_env.task.get_termination=original_term
        valid=trajectory[25:] if len(trajectory)>25 else trajectory
        occupancy=sum(x['success'] for x in trajectory if x['step']>25)/max(1,steps-25)
        errors=[x['error_in_native_tolerances'] for x in valid]
        return dict(case=case,task=self.task,arm=self.arm,level=level,latency_enabled=latency,control_dt=self.dt,feedback_steps=feedback,steps=len(trajectory),ever_success=hit is not None and not collision,first_hit_step=hit,final_success=bool(trajectory[-1]['success']) and not collision,occupancy_after25=occupancy,mean_normalized_error=float(np.mean(errors)),collision=bool(collision),waiting_steps=waiting,plans=plans,trajectory=trajectory)
