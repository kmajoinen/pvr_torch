"""
Empirical check for mode=pvr_ft's share_features_extractor=True routing.
Standalone diagnostic, not part of the training pipeline -- run on roihu
where stable-baselines3 + the sim backends are actually installed:

    python check_grad_routing.py [env_id] [embedding_name]

Print 1: does each of actor.optimizer / critic.optimizer actually contain
the features_extractor params it's supposed to (per SB3 2.9.0 source:
share_features_extractor=True means actor.features_extractor is
critic.features_extractor is the ONLY copy that lives in an optimizer at
all -- critic.forward() wraps its own use of it in
th.set_grad_enabled(False), so no gradient reaches it from the critic side,
but the same nn.Module's params are still registered under
model.policy.parameters() -> model.actor.optimizer via SB3's normal
"actor optimizer owns policy.parameters()" setup).

Print 2: does the shared copy's weight sum actually move after ~50
finetune gradient steps, and does critic_target's independent copy NOT
move except via polyak (tau), confirming which side is live.
"""
import sys

from stable_baselines3 import SAC

from train_sac import make_pixel_env
from feature_extractor import PVRFeaturesExtractor

env_id = sys.argv[1] if len(sys.argv) > 1 else "dm_control/cheetah-run-v0"
embedding_name = sys.argv[2] if len(sys.argv) > 2 else "resnet18"

env = make_pixel_env(env_id)

policy_kwargs = dict(
    features_extractor_class=PVRFeaturesExtractor,
    features_extractor_kwargs=dict(embedding_name=embedding_name, freeze=False, disable_cuda=False),
    net_arch=[256, 256],
    normalize_images=False,
    share_features_extractor=True,
)
model = SAC(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    buffer_size=2000,
    learning_starts=300,
    batch_size=64,
    train_freq=1,
    gradient_steps=1,
    verbose=0,
)

actor_fe = model.policy.actor.features_extractor
critic_fe = model.policy.critic.features_extractor
target_fe = model.policy.critic_target.features_extractor


def param_ids(module):
    return {id(p) for p in module.parameters()}


def optimizer_ids(opt):
    ids = set()
    for group in opt.param_groups:
        ids.update(id(p) for p in group["params"])
    return ids


actor_ids = param_ids(actor_fe)
critic_ids = param_ids(critic_fe)
target_ids = param_ids(target_fe)
actor_opt_ids = optimizer_ids(model.actor.optimizer)
critic_opt_ids = optimizer_ids(model.critic.optimizer)

print("=== identity ===")
print("actor.features_extractor is critic.features_extractor:  ", actor_fe is critic_fe)
print("critic.features_extractor is critic_target.features_extractor:", critic_fe is target_fe)

print()
print("=== print 1: which optimizer(s) hold each copy's params ===")
print("actor_fe  params subset-of actor.optimizer: ", actor_ids <= actor_opt_ids)
print("actor_fe  params subset-of critic.optimizer:", actor_ids <= critic_opt_ids)
print("critic_fe params subset-of actor.optimizer: ", critic_ids <= actor_opt_ids)
print("critic_fe params subset-of critic.optimizer:", critic_ids <= critic_opt_ids)
print("target_fe params subset-of actor.optimizer: ", target_ids <= actor_opt_ids)
print("target_fe params subset-of critic.optimizer:", target_ids <= critic_opt_ids)


def weight_sum(module):
    return sum(p.detach().float().sum().item() for p in module.parameters())


# Fill the replay buffer via random-policy rollout only -- learning_starts=300
# means SB3's off-policy loop won't call train() during this .learn() call
# (it only trains once num_timesteps > learning_starts).
model.learn(total_timesteps=model.learning_starts, log_interval=1000)

before = dict(actor=weight_sum(actor_fe), critic=weight_sum(critic_fe), target=weight_sum(target_fe))
model.train(gradient_steps=50, batch_size=model.batch_size)
after = dict(actor=weight_sum(actor_fe), critic=weight_sum(critic_fe), target=weight_sum(target_fe))

print()
print("=== print 2: weight-sum movement after 50 finetune gradient steps ===")
for key in before:
    delta = after[key] - before[key]
    print(f"{key:8s} before={before[key]:.6f} after={after[key]:.6f} delta={delta:.6e} moved={abs(delta) > 1e-8}")
