import gymnasium as gym
import numpy as np


class RepeatAction(gym.Wrapper):
    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, done, truncated,info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if done or truncated:
                break

        return obs, total_reward, done,truncated, combined_info
