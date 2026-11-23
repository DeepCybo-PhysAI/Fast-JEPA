
import stable_worldmodel as swm

class SeededWorld(swm.World):
    def __init__(self, *args, diagnostic_seed, **kwargs):
        self.diagnostic_seed=int(diagnostic_seed)
        self.diagnostic_reset_calls=[]
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            raise ValueError('This diagnostic expects missing dataset seeds; refuse to override existing ones')
        actual=[self.diagnostic_seed+i for i in range(self.num_envs)]
        self.diagnostic_reset_calls.append(actual)
        return super().reset(seed=actual, options=options)
