from gym.envs.registration import register


register(
    id='gymconnectx/ConnectGameEnv',
    entry_point='gymconnectx.envs:ConnectGameEnv',
)