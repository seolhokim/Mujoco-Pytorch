import gym
from gym.spaces import Discrete, Box
from gym.core import Env

import numpy as np

class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None):
        """
        """
        super().__init__(env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, done, info)

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps
class GymEnv(Env):
    """
    Taken from rllab gym_env.py
    """

    def __init__(self, env_name, video=False,
                 log_dir=None,
                 force_reset=False,
                 is_atari=False,
                 *args, **kwargs):

        self.env = env = gym.envs.make(env_name)
        self._original_env = env

        if is_atari:
            self.env = wrap_atari(env)
            # from yarlp.utils.wrap_atari import wrap_deepmind, wrap_deepmind2
            # self.env = wrap_deepmind2(env_name)
        else:
            self.env = MonitorEnv(env)

        assert isinstance(video, bool)
        if log_dir is None:
            self.monitoring = False
        else:
            if not video:
                video_schedule = NoVideoSchedule()
            else:
                video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(
                self.env, log_dir, video_callable=video_schedule,
                force=True)
            self.monitoring = True

        self.env_id = env.spec.id
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def action_space(self):
        return self.env.action_space

    @staticmethod
    def env_action_space_is_discrete(env):
        if isinstance(env.action_space, Discrete):
            return True
        elif isinstance(env.action_space, Box):
            return False
        else:
            raise NotImplementedError('Uknown base environment: ', env)

    @staticmethod
    def get_env_action_space_dim(env):
        if GymEnv.env_action_space_is_discrete(env):
            return env.action_space.n
        return env.action_space.shape[0]

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        if self._force_reset and self.monitoring:
            assert isinstance(self.env, gym.wrappers.Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self._original_env.close()

    def seed(self, i=None):
        return self.env.seed(i)

    @property
    def spec(self):
        return self.env.spec

    def __str__(self):
        return "GymEnv: %s" % self.env

    @property
    def unwrapped(self):
        return self.env.unwrapped


class NormalizedGymEnv(GymEnv):
    """
    Taken from rllab normalized_env.py
    """

    def __init__(self, env_name,
                 video=False,
                 log_dir=None,
                 force_reset=False,
                 scale_reward=1.,
                 min_reward_std=1e-2,
                 min_obs_std=1e-2,
                 norm_obs_clip=5,
                 normalize_obs=False,
                 normalize_rewards=False,
                 scale_continuous_actions=False,
                 is_atari=False,
                 *args, **kwargs):
        super().__init__(env_name=env_name, video=video,
                         log_dir=log_dir, force_reset=force_reset,
                         is_atari=is_atari, *args, **kwargs)
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_rewards = normalize_rewards
        self._scale_continuous_actions = scale_continuous_actions
        self.is_atari = is_atari

        if normalize_obs is True:
            assert is_atari is False,\
                "normalize_obs must be False if is_atari is True"
            self._obs_rms = RunningMeanStd(
                shape=(self.env.observation_space.shape),
                min_std=min_obs_std, clip_val=norm_obs_clip)

        if normalize_rewards is True:
            self._reward_rms = RunningMeanStd(
                shape=(1), min_std=min_reward_std)

    @property
    def action_space(self):
        if isinstance(self.env.action_space, Box):
            ub = np.ones(self.env.action_space.shape)
            return Box(-1 * ub, ub)
        return self.env.action_space

    def _update_rewards(self, r, done):
        self._reward_rms.cache(r)
        r = self._reward_rms.normalize(r)
        if done:
            self._reward_rms.update()
        return r

    def _update_obs(self, obs, done):
        self._obs_rms.cache(obs)
        obs = self._obs_rms.normalize(obs)
        if done:
            self._obs_rms.update()
        return obs

    def reset(self):
        ob = super().reset()
        if self._normalize_obs:
            return self._update_obs(ob, False)
        return ob

    def step(self, action):
        if self._scale_continuous_actions:
            if isinstance(self.env.action_space, Box):
                # rescale the action
                lb, ub = self.env.action_space.low, self.env.action_space.high
                scaled_action = lb + (action[0] + 1.) * 0.5 * (ub - lb)
                scaled_action = np.clip(scaled_action, lb, ub)
                action = scaled_action

        wrapped_step = self.env.step(action)
        next_obs, reward, done, info = wrapped_step

        if self._normalize_obs:
            next_obs = self._update_obs(next_obs, done)

        if self._normalize_rewards:
            reward = self._update_rewards(reward, done)

        return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized GymEnv: %s" % self.env


class RunningMeanStd(object):
    """
    RunningMeanStd
    """

    def __init__(self, shape, min_std=1e-6, clip_val=None):
        self._min_std = min_std
        self._clip_val = clip_val
        self._cache = []
        self._mean = np.zeros(shape)
        self._std = np.ones(shape)
        self._count = 0.

    def normalize(self, x):
        xn = (x - self._mean) / self._std
        if self._clip_val:
            xn = np.clip(xn, -self._clip_val, self._clip_val)

        if np.isscalar(x):
            return np.asscalar(xn)

        return xn

    def cache(self, x):
        self._cache.append(x)

    def update(self):
        X = np.array(self._cache)
        if X.shape[0] <= 1:
            # wait for more data to avoid numerical errors in std calc
            return

        avg_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0, ddof=1)
        if self._count == 0:
            self._std = np.clip(std_X, self._min_std, None)
            self._mean = avg_X
            self._count += X.shape[0]
        else:
            delta = avg_X - self._mean
            m_a = np.square(self._std) * (self._count - 1)
            m_b = np.square(std_X) * (X.shape[0] - 1)
            M2 = m_a + m_b + delta ** 2 * self._count * X.shape[0] /\
                (self._count + X.shape[0])
            M2 = np.sqrt(M2 / (self._count + X.shape[0] - 1))
            self._std = np.clip(M2, self._min_std, None)
            self._count += X.shape[0]
            self._mean = self._mean + delta * X.shape[0] / self._count
        self._cache = []