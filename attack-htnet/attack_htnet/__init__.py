from gym.envs.registration import register

register(
    id='htnet-v0',
    entry_point='attack_htnet.envs:EhrEnv',
)
