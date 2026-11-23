
import torch
from torch import nn
from vendor.fast_lewm_module import ActionPrefixEmbedder, ARPredictor


class FastWorldModel(nn.Module):
    def __init__(self, latent_dim, action_dim, transformer_depth=3,
                 transformer_heads=6, transformer_dim_head=32,
                 transformer_mlp_dim=768, depth=6, mlp_dim=2048,
                 value_heads=16, value_dim_head=64, dropout=0.1,
                 action_fusion_hidden_dim=768):
        super().__init__()
        self.latent_dim, self.action_dim = int(latent_dim), int(action_dim)
        self.prefix_encoder = ActionPrefixEmbedder(
            input_dim=self.action_dim, emb_dim=self.latent_dim,
            transformer_depth=transformer_depth, transformer_heads=transformer_heads,
            transformer_dim_head=transformer_dim_head,
            transformer_mlp_dim=transformer_mlp_dim,
            use_latent_condition=True, latent_dim=self.latent_dim,
            use_positional_encoding=True,
        )
        self.transition = ARPredictor(
            input_dim=self.latent_dim, hidden_dim=self.latent_dim,
            output_dim=self.latent_dim, depth=depth, mlp_dim=mlp_dim,
            value_heads=value_heads, value_dim_head=value_dim_head,
            dropout=dropout, emb_dropout=0.0, token_processing='batch',
            action_fusion_hidden_dim=action_fusion_hidden_dim,
            action_fusion_zero_init=True,
        )

    def forward(self, initial, actions, last_only=False):
        if initial.ndim == 2:
            initial = initial[:, None]
        if initial.ndim != 3 or initial.shape[1:] != (1, self.latent_dim):
            raise ValueError('Expected exactly one initial latent [B,1,D]')
        if (actions.ndim != 3 or actions.shape[0] != initial.shape[0]
                or actions.shape[-1] != self.action_dim or actions.shape[1] < 1):
            raise ValueError('Expected normalized action blocks [B,H,action_dim]')
        tokens = self.prefix_encoder(actions, latent=initial, return_last_only=last_only)

        return self.transition(initial, tokens)


def fast_prediction_loss(model, embeddings, actions):

    horizon = min(model.fast_horizon, embeddings.shape[1] - 1, actions.shape[1])
    if horizon < 1:
        raise ValueError('Fast supervision needs at least one transition')
    predictions = model.predict_prefix(embeddings[:, :1], actions[:, :horizon])
    targets = embeddings[:, 1:horizon + 1]
    if model.fast_target_stop_gradient:
        targets = targets.detach()
    per_horizon = (predictions - targets).square().mean(dim=(0, 2))
    return per_horizon.mean(), {f'wm_mse_h{k+1}': v for k, v in enumerate(per_horizon)}
