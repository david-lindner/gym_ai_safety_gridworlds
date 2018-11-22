import importlib
import random
import gym

from gym.utils import seeding
from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds_viewer.view_agent import AgentViewer


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

    def __init__(self, env_name, pause=0.1):
        self._env_name = env_name
        self._pause = pause
        self._viewer = None
        self._env = factory.get_environment_obj(env_name)
        self.action_space = GridworldsActionSpace(self._env)
        self.observation_space = GridworldsObservationSpace(self._env)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    def step(self, action):
        timestep = self._env.step(action)
        obs = timestep.observation
        reward = 0.0 if timestep.reward is None else timestep.reward
        done = timestep.step_type.last()
        return (obs, reward, done, {})

    def reset(self):
        timestep = self._env.reset()
        if self._viewer is not None:
            self._viewer.reset_time()

        return timestep.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="human", close=False):
        if close and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        elif close:
            pass
        elif self._viewer is None:
            self._viewer = init_viewer(self._env_name, self._pause)
            self._viewer.display(self._env)
        else:
            print("render 4")
            self._viewer.display(self._env)


class GridworldsActionSpace(gym.Space):
    def __init__(self, env):
        action_spec = env.action_spec()
        assert action_spec.name == "discrete"
        assert action_spec.dtype == "int32"
        assert len(action_spec.shape) == 1 and action_spec.shape[0] == 1
        self.min_action = action_spec.minimum
        self.max_action = action_spec.maximum
        super(self.__class__, self).__init__(
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
        super(self.__class__, self)

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
        return observation

    def contains(self, x):
        result = True
        for key, spec in self.observation_spec_dict.items():
            if spec != {}:  # no specification restrics this observation
                if key in x.keys():
                    try:
                        spec.validate(x[key])
                    except ValueError:
                        result = False
                        break
                else:
                    return False
        return result


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
