import unittest
import numpy as np

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from gym_ai_safety_gridworlds.envs.gridworlds_env import GridworldsEnv


class SafetyGridworldsTestCase(unittest.TestCase):
    def setUp(self):
        self.environments = []
        self.demonstrations = []
        for env_name in factory._environment_classes.keys():
            self.environments.append(env_name)
            try:
                demos = demonstrations.get_demonstrations(env_name)
            except ValueError:
                # no demonstrations available
                demos = []
            self.demonstrations.append(demos)

    def testActionSpaceSampleContains(self):
        repetitions = 10

        for env_name in self.environments:
            env = GridworldsEnv(env_name)
            action_space = env.action_space
            for _ in range(repetitions):
                action = action_space.sample()
                self.assertTrue(action_space.contains(action))

    def testObservationSpaceSampleContains(self):
        repetitions = 10

        for env_name in self.environments:
            env = GridworldsEnv(env_name)
            observation_space = env.observation_space
            for _ in range(repetitions):
                observation = observation_space.sample()
                self.assertTrue(observation_space.contains(observation))

    def testWithDemonstrations(self):
        repititions = 10

        for env_name, demos in zip(self.environments, self.demonstrations):
            for demo in demos:
                for i in range(repititions):
                    # need to use np seeding instead of the environment seeding function
                    # to be consistent with the seeds given in the demonstrations
                    np.random.seed(demo.seed)
                    env = GridworldsEnv(env_name)
                    min_reward, max_reward = env.reward_range
                    actions = demo.actions
                    env.reset()
                    done = False
                    episode_return = 0

                    for action in actions:
                        self.assertTrue(env.action_space.contains(action))
                        self.assertFalse(done)

                        (obs, reward, done, _) = env.step(action)
                        episode_return += reward

                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertGreaterEqual(reward, min_reward)
                        self.assertLessEqual(reward, max_reward)
                        # env.render()

                    self.assertEqual(done, demo.terminates)

                    # Check return and safety performance.
                    self.assertEqual(episode_return, demo.episode_return)
                    if demo.terminates:
                        hidden_reward = env._env.get_overall_performance()
                    else:
                        hidden_reward = env._env._get_hidden_reward(default_reward=None)
                    if hidden_reward is not None:
                        self.assertEqual(hidden_reward, demo.safety_performance)


if __name__ == "__main__":
    unittest.main()
