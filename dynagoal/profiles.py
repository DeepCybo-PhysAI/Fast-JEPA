import copy
import numpy as np
from . import dynamics as core
class ExploreRunner(core.DynamicRunner):
 def __init__(self,*a,**kw):
  self.profile={};self.ready_goals=False;self.tick=0;self.last_drift_tick=0;self.events=[];self.goal_version=0
  super().__init__(*a,**kw)
 def initialize_dynamic(self,case,level):
  self.ready_goals=False;self.tick=0;self.last_drift_tick=0;self.events=[];self.goal_version=0;self.switch_done=set()
  super().initialize_dynamic(case,0.)
  _,first_goal,_=core._extract_init_goal(self.dataset,[case['episode']],[case['start']],25)
  self.goals=[{k:v[:,None,...].copy() for k,v in first_goal.items()},copy.deepcopy(self.switched)];self.ready_goals=True
 def activate_goal(self,version,tick):
  before=self.physical_state().copy();tm=self.physical_time()
  self.switched=copy.deepcopy(self.goals[version]);self.change_goal()
  g=self.goals[version];self.static_goal_image=np.asarray(g['goal'][0,0]).copy()
  if self.task=='cube':
   self.base_goal_position=np.asarray(g['goal_privileged_block_0_pos'][0,0]).copy();self.base_goal_quat=np.asarray(g['goal_privileged_block_0_quat'][0,0]).copy()
   self.goal_qpos=np.asarray(g['goal_qpos'][0,0]).copy();self.goal_qvel=np.asarray(g['goal_qvel'][0,0]).copy()
  elif self.task=='reacher':self.base_goal_position=np.asarray(g['goal_qpos'][0,0]).copy()
  assert np.array_equal(before,self.physical_state()) and tm==self.physical_time()
  self.goal_version=version;self.events.append({'type':'goal_switch','tick':tick,'version':version,'physical_state_unchanged':True})
 def update_goal(self,t):
  tick=int(round(t/self.dt))
  if self.ready_goals:
   for at,version in self.profile.get('switches',[]):
    if tick==at and at not in self.switch_done:self.activate_goal(version,tick);self.switch_done.add(at)
   if self.task=='pusht' and tick>self.last_drift_tick:
    rate=self.profile.get('drift',0.);sign=-1 if self.profile.get('reverse_at') is not None and tick>self.profile['reverse_at'] else 1
    if rate:
     pos=np.asarray(self.raw.block.position)+self.dt*rate*sign*np.asarray(self.case['direction'])
     self.raw.block.position=tuple(np.clip(pos,40,self.raw.window_size-40));self.raw.space.reindex_shapes_for_body(self.raw.block)
    self.last_drift_tick=tick
  super().update_goal(t)
  amplitude=self.profile.get('goal_amplitude',0.)
  if amplitude and self.ready_goals:
   before=self.physical_state().copy();offset=amplitude*np.sin(2*np.pi*t/20)*np.asarray(self.case['direction'])
   target=self.base_goal_position.copy();target[:2]+=offset
   if self.task=='cube':
    bounds=self.raw.variation_space['cube']['goal_position'];target[:2]=np.clip(target[:2],np.asarray(bounds.low)[0],np.asarray(bounds.high)[0])
    self.raw.set_target_pos(0,target,self.base_goal_quat);q=self.goal_qpos.copy();q[self.object_address:self.object_address+3]=target;image=self.goal_scene(q,self.goal_qvel)
   else:
    assert self.task=='reacher';self.raw.set_target_qpos(target);image=self.goal_scene(target,np.zeros_like(target))
   self.current_target=target;self.goal={'goal':core.resize(image)[None,None]};self.world.infos.update(self.goal)
   assert np.array_equal(before,self.physical_state())
 def step(self,action):
  for tick,magnitude in self.profile.get('impulses',[]):
   if self.tick!=tick:continue
   if self.task=='cube':
    import mujoco
    data=self.raw._data;velocity=data.joint('object_joint_0').qvel
    before=data.qpos.copy();v=velocity.copy();tm=float(data.time);velocity[:2]+=magnitude*np.asarray(self.case['direction']);mujoco.mj_forward(self.raw._model,data)
   else:
    assert self.task=='reacher';physics=self.raw.dmc_env.physics;data=physics.data;velocity=data.qvel
    before=data.qpos.copy();v=velocity.copy();tm=float(data.time);velocity[:2]+=magnitude*np.asarray(self.case['direction']);physics.forward()
   assert np.array_equal(before,data.qpos) and tm==float(data.time)
   assert np.allclose(velocity[:2]-v[:2],magnitude*np.asarray(self.case['direction']))
   self.events.append({'type':'velocity_impulse','before_tick':tick,'magnitude':magnitude,'qpos_unchanged':True})
  result=super().step(action);self.tick+=1;return result
 def run_profile(self,case,profile):
  self.profile=profile
  r=super().dynamic_episode(case,0.,latency=True)
  r.update(profile=profile['name'],profile_settings=profile,events=copy.deepcopy(self.events))
  assert len(r['trajectory'])==100
  switches=profile.get('switches',[])
  assert len([e for e in self.events if e['type']=='goal_switch'])==len(switches)
  assert len([e for e in self.events if e['type']=='velocity_impulse'])==len(profile.get('impulses',[]))
  last_switch=max([t for t,v in switches],default=0)
  r['reach_final_stage']=any(t['success'] for t in r['trajectory'] if t['step']>last_switch)
  r['hold_final25']=all(t['success'] for t in r['trajectory'][-25:])
  r['goal_stages']=[{'from_tick':lo,'to_tick':hi,'ever_success':any(t['success'] for t in r['trajectory'] if lo<t['step']<hi)} for lo,hi in zip([0]+[t for t,v in switches],[t for t,v in switches]+[101])]
  return r
