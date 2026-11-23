
import torch
from einops import rearrange
from jepa import JEPA as SharedJEPA
from fast_world_model import FastWorldModel


class JEPA(SharedJEPA):
    def __init__(self, *args, world_model_type='fast', fast_horizon=5,
                 fast_target_stop_gradient=False, fast_kwargs=None, **kwargs):
        if world_model_type != 'fast':
            raise ValueError("Fast-JEPA only supports world_model_type='fast'")
        # Hydra constructs the configured placeholder predictor before this call.
        # Keep that initialization: removing it changes the RNG seen by the Actor.
        super().__init__(*args, **kwargs)
        self.world_model_type = world_model_type
        self.fast_horizon = int(fast_horizon)
        self.fast_target_stop_gradient = bool(fast_target_stop_gradient)
        if self.fast_horizon < 1:
            raise ValueError('fast_horizon must be positive')
        if self.predict_residual:
            raise ValueError('Fast-JEPA predicts absolute latents; disable residual mode')
        latent_dim = self.intent_actor.embed_dim

        with torch.random.fork_rng(devices=[]):
            self.predictor = FastWorldModel(latent_dim, self.get_action_dim(), **dict(fast_kwargs or {}))

    def predict_prefix(self, initial, actions, last_only=False):
        prediction = self.predictor(initial, actions, last_only=last_only)
        b, h, _ = prediction.shape
        return self.pred_proj(prediction.reshape(b*h, -1)).reshape(b, h, -1)

    def predict(self, embedding, action_embedding):
        raise RuntimeError('Fast-JEPA prediction requires raw action prefixes; use predict_prefix')

    def get_cost(self, info, action_candidates):

        if action_candidates.ndim != 4 or action_candidates.shape[-1] != self.get_action_dim():
            raise ValueError('Candidates must be [B,S,H,normalized_action_block_dim]')
        device = next(self.parameters()).device
        b, samples, h, a = action_candidates.shape
        collapsed = {k: v[:, 0].to(device) for k, v in info.items() if torch.is_tensor(v)}
        collapsed.pop('action', None)
        initial = self.encode(dict(collapsed))['emb'][:, -1:]
        goal = self._encode_goal(collapsed)
        initial = initial[:, None].expand(-1, samples, -1, -1).reshape(b*samples, 1, -1)

        predicted = self.predict_prefix(initial, action_candidates.to(device).reshape(b*samples, h, a), last_only=True)
        predicted = predicted.reshape(b, samples, -1)
        return (predicted - goal[:, None].detach()).square().sum(-1)

    def rollout(self, info, action_sequence, history_size=None):
        raise NotImplementedError('Fast-JEPA uses terminal prefix scoring via get_cost; recursive rollout is unsupported')

    @torch.inference_mode()
    def get_action(self, info, horizon=1, prefix_actions=None):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
        initial = {k: v for k, v in info.items() if torch.is_tensor(v)}
        b = initial['pixels'].shape[0]
        actions_history = self._coerce_action_history(
            initial.get('action'), batch_size=b, seq_len=initial['pixels'].shape[1],
            action_dim=self.get_action_dim(), device=device, dtype=dtype)
        initial['action'] = actions_history
        anchor = self.encode(initial)['emb'][:, -1:]
        goal = self._encode_goal(info)
        current = anchor[:, 0]
        previous = actions_history[:, -1:]
        prefix = []
        if prefix_actions is not None and prefix_actions.numel():
            if (prefix_actions.ndim != 3 or prefix_actions.shape[0] != b
                    or prefix_actions.shape[-1] != self.get_action_dim()):
                raise ValueError('prefix_actions must contain complete action blocks [B,K,A]')
            prefix = list(prefix_actions.to(device=device, dtype=dtype).unbind(1))
            current = self.predict_prefix(anchor, torch.stack(prefix, dim=1), last_only=True)[:, -1]
            previous = prefix[-1][:, None]
        plan, norms = [], []
        for step in range(horizon):
            intent = self.goal_intent(current, goal, horizon-step)
            previous_embedding = self.action_encoder(previous)[:, -1]
            action = self.intent_actor.action_mean(current, intent, previous_embedding)
            plan.append(action)
            prefix.append(action)
            norms.append(intent.norm(dim=-1).mean())


            current = self.predict_prefix(anchor, torch.stack(prefix, dim=1), last_only=True)[:, -1]
            previous = action[:, None]
        self.last_direct_diagnostics = {
            'intent_norm': float(torch.stack(norms).mean()),
            'terminal_latent_error': float((current-goal).norm(dim=-1).mean()),
            'forward_calls': float(horizon + bool(prefix_actions is not None and prefix_actions.numel())),
            'candidate_sequences': 0.0,
        }
        return torch.stack(plan, dim=1)
