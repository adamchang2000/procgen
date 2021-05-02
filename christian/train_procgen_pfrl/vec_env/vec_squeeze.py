from .vec_env import VecEnvObservationWrapper


class VecSqueezeObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        # if obs[self.key].shape[0] > 1:
        #     print(obs[self.key].shape)
        obs_shape = obs[self.key].shape
        if len(obs_shape) > 4:
            print("VecSqueezeObs_process -> obs_shape:", obs_shape)
            first_axis_num = obs[self.key].shape[0]
            assert first_axis_num == 1, \
                f'Invalid Squeeze Shape: {obs[self.key].shape}'
            return obs[self.key].squeeze(0)

        assert len(obs_shape) == 4, f"Invalid Obs Shape: {obs_shape}"
        return obs[self.key]
        