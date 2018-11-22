import importlib
import random
import gym

from gym import error
from gym.utils import seeding
from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds_viewer.view_agent import AgentViewer

INFO_HIDDEN_REWARD = "hidden_reward"
INFO_OBSERVED_REWARD = "observed_reward"
INFO_DISCOUNT = "discount"


class GridworldsEnv(gym.Env):
    """
    An unofficial OpenAI Gym interface for DeepMind ai-safety-gridworlds

    This class implement OpenAI Gym interface for the ai-safety-gridworlds
    of DeepMind. OpenAI Gym has become a standard interface to a collection of
    environments for reinforcement learning (RL). By providing a Gym interface,
    it helps researchers in the field of AI safety to compare RL algorithms
    (including existing implementations such as OpenAI Baselines) on DeepMind
    ai-safety-gridworlds.
    """

    metadata = {
        # TODO
        "render.modes": ["human"],
        "video.frames_per_second": 50,
    }

    def __init__(self, env_name, cheat=False, render_animation_delay=0.1):
        self._env_name = env_name
        self.cheat = cheat
        self._render_animation_delay = render_animation_delay
        self._viewer = None
        self._env = factory.get_environment_obj(env_name)
        self._rbg = None
        self._last_hidden_reward = 0
        self.action_space = GridworldsActionSpace(self._env)
        self.observation_space = GridworldsObservationSpace(self._env)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def step(self, action):
        timestep = self._env.step(action)
        obs = timestep.observation
        self._rgb = obs["RGB"]

        reward = 0.0 if timestep.reward is None else timestep.reward
        done = timestep.step_type.last()

        cumulative_hidden_reward = self._env._get_hidden_reward(default_reward=None)
        if cumulative_hidden_reward is not None:
            hidden_reward = cumulative_hidden_reward - self._last_hidden_reward
            self._last_hidden_reward = cumulative_hidden_reward
        else:
            hidden_reward = None

        info = {
            INFO_HIDDEN_REWARD: hidden_reward,
            INFO_OBSERVED_REWARD: reward,
            INFO_DISCOUNT: timestep.discount,
        }

        for k, v in obs.items():
            if k not in ("board", "RGB"):
                info[k] = v

        if self.cheat:
            if hidden_reward is None:
                error.Error("gridworld '%s' does not support cheating" % self._env_name)
                return_reward = reward
                self.cheat = False
            else:
                return_reward = hidden_reward
        else:
            return_reward = reward

        return (obs["board"], return_reward, done, info)

    def reset(self):
        timestep = self._env.reset()
        self._rgb = timestep.observation["RGB"]
        if self._viewer is not None:
            self._viewer.reset_time()

        return timestep.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human"):
        if mode == "rgb_array":
            if self._rgb is None:
                error.Error("environment has to be reset before rendering")
            else:
                return self._rgb
        elif mode == "ansi":
            if self._env._current_game is None:
                error.Error("environment has to be reset before rendering")
            else:
                ascii_np_array = self._env._current_game._board.board
                ansi_string = "\n".join(
                    [
                        " ".join([chr(i) for i in ascii_np_array[j]])
                        for j in range(ascii_np_array.shape[0])
                    ]
                )
                return ansi_string
        elif mode is "human":
            if self._viewer is None:
                self._viewer = init_viewer(self._env_name, self._render_animation_delay)
                self._viewer.display(self._env)
            else:
                self._viewer.display(self._env)
        else:
            super(GridworldsEnv, self).render(mode=mode)  # just raise an exception


class GridworldsActionSpace(gym.Space):
    def __init__(self, env):
        action_spec = env.action_spec()
        assert action_spec.name == "discrete"
        assert action_spec.dtype == "int32"
        assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
        self.min_action = action_spec.minimum
        self.max_action = action_spec.maximum
        super(GridworldsActionSpace, self).__init__(
            shape=action_spec.shape, dtype=action_spec.dtype
        )

    def sample(self):
        return random.randint(self.min_action, self.max_action)

    def contains(self, x):
        """
        Return True is x is a valid action. Note, that this does not use the
        pycolab validate function, because that expects a numpy array and not
        an individual action.
        """
        return self.min_action <= x <= self.max_action


class GridworldsObservationSpace(gym.Space):
    def __init__(self, env):
        self.observation_spec_dict = env.observation_spec()
        super(GridworldsObservationSpace, self)

    def sample(self):
        """
        Use pycolab to generate an example observation. Note that this is not a
        random sample, but might return the same observation for every call.
        """
        observation = {}
        for key, spec in self.observation_spec_dict.items():
            if spec == {}:
                observation[key] = {}
            else:
                observation[key] = spec.generate_value()
        return observation["board"]

    def contains(self, x):
        if "board" in self.observation_spec_dict.keys():
            try:
                self.observation_spec_dict["board"].validate(x)
                return True
            except ValueError:
                return False
        else:
            return False


def init_viewer(env_name, pause):
    (color_bg, color_fg) = get_color_map(env_name)
    av = AgentViewer(pause, color_bg=color_bg, color_fg=color_fg)
    return av


def get_color_map(env_name):
    module_prefix = "ai_safety_gridworlds.environments."
    env_module_name = module_prefix + env_name
    env_module = importlib.import_module(env_module_name)
    color_bg = env_module.GAME_BG_COLOURS
    color_fg = env_module.GAME_FG_COLOURS
    return (color_bg, color_fg)
