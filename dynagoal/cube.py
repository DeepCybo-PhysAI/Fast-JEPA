import numpy as np
from . import profiles as ex
from . import deployment as dep
class CubeRunner(ex.ExploreRunner):
 def initialize_dynamic(self,case,level):
  super().initialize_dynamic(case,level)
  _,initial_goal,_=dep._extract_init_goal(self.dataset,[case['episode']],[case['start']],0)
  self.goals.append({k:v[:,None,...].copy() for k,v in initial_goal.items()})
 def update_goal(self,t):
  profile=self.profile
  period=profile.get('goal_period_ticks')
  if period is None:return super().update_goal(t)
  self.profile={**profile,'goal_amplitude':0.}
  try:super().update_goal(t)
  finally:self.profile=profile
  if not self.ready_goals:return
  before=self.physical_state().copy();tm=self.physical_time()
  target=self.base_goal_position.copy();target[:2]+=profile['goal_amplitude']*np.sin(2*np.pi*(t/self.dt)/period)*np.asarray(self.case['direction'])
  bounds=self.raw.variation_space['cube']['goal_position'];target[:2]=np.clip(target[:2],np.asarray(bounds.low)[0],np.asarray(bounds.high)[0])
  self.raw.set_target_pos(0,target,self.base_goal_quat);q=self.goal_qpos.copy();q[self.object_address:self.object_address+3]=target
  self.current_target=target;self.goal={'goal':ex.core.resize(self.goal_scene(q,self.goal_qvel))[None,None]};self.world.infos.update(self.goal)
  assert np.array_equal(before,self.physical_state()) and tm==self.physical_time()
