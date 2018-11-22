import unittest
import numpy as np

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from gym_ai_safety_gridworlds.envs.gridworlds_env import (
    GridworldsEnv,
    INFO_HIDDEN_REWARD,
    INFO_OBSERVED_REWARD,
)
from ai_safety_gridworlds.environments.shared.safety_game import Actions


class SafetyGridworldsTestCase(unittest.TestCase):
    def _check_rgb(self, rgb_list):
        first_shape = rgb_list[0].shape
        for rgb in rgb_list:
            self.assertEqual(len(rgb.shape), 3)
            self.assertEqual(rgb.shape[0], 3)
            self.assertEqual(rgb.shape, first_shape)

    def _check_ansi(self, ansi_list):
        first_len = len(ansi_list[0])
        first_newline_count = ansi_list[0].count("\n")
        for ansi in ansi_list:
            self.assertEqual(len(ansi), first_len)
            self.assertEqual(ansi.count("\n"), first_newline_count)

    def _check_reward(self, env, reward):
        min_reward, max_reward = env.reward_range
        self.assertGreaterEqual(reward, min_reward)
        self.assertLessEqual(reward, max_reward)

    def _check_action_observation_valid(self, env, action, observation):
        self.assertTrue(env.action_space.contains(action))
        self.assertTrue(env.observation_space.contains(observation))

    def _check_rewards(
        self,
        env,
        cheat,
        demo,
        epsiode_info_observed_return,
        episode_info_hidden_return,
        episode_return,
    ):
        # check observed and hidden rewards
        self.assertEqual(epsiode_info_observed_return, demo.episode_return)

        hidden_reward = env._env._get_hidden_reward(default_reward=None)

        if hidden_reward is not None:
            self.assertEqual(episode_info_hidden_return, demo.safety_performance)
            self.assertEqual(episode_info_hidden_return, hidden_reward)

        if cheat and hidden_reward is not None:
            self.assertEqual(episode_info_hidden_return, episode_return)
        else:
            self.assertEqual(epsiode_info_observed_return, episode_return)

    def setUp(self):
        self.demonstrations = {}
        for env_name in factory._environment_classes.keys():
            try:
                demos = demonstrations.get_demonstrations(env_name)
            except ValueError:
                # no demonstrations available
                demos = []
            self.demonstrations[env_name] = demos

        # add demo that fails
        self.demonstrations["absent_supervisor"].append(
            demonstrations.Demonstration(0, [Actions.DOWN] * 3, 47, 17, True)
        )

    def testActionSpaceSampleContains(self):
        repetitions = 10

        for env_name in self.demonstrations.keys():
            env = GridworldsEnv(env_name)
            action_space = env.action_space
            for _ in range(repetitions):
                action = action_space.sample()
                self.assertTrue(action_space.contains(action))

    def testObservationSpaceSampleContains(self):
        repetitions = 10

        for env_name in self.demonstrations.keys():
            env = GridworldsEnv(env_name)
            observation_space = env.observation_space
            for _ in range(repetitions):
                observation = observation_space.sample()
                self.assertTrue(observation_space.contains(observation))

    def testWithDemonstrations(self):
        repititions = 10

        for env_name, demos in self.demonstrations.items():
            for demo in demos:
                for i in range(repititions):
                    for cheat in (True, False):
                        # need to use np seed instead of the environment seed function
                        # to be consistent with the seeds given in the demonstrations
                        np.random.seed(demo.seed)
                        env = GridworldsEnv(env_name, cheat=cheat)

                        actions = demo.actions
                        env.reset()
                        done = False

                        episode_return = 0
                        epsiode_info_observed_return = 0
                        episode_info_hidden_return = 0

                        rgb_list = [env.render("rgb_array")]
                        ansi_list = [env.render("ansi")]

                        for action in actions:
                            self.assertFalse(done)

                            (obs, reward, done, info) = env.step(action)
                            episode_return += reward
                            epsiode_info_observed_return += info[INFO_OBSERVED_REWARD]

                            if info[INFO_HIDDEN_REWARD] is not None:
                                episode_info_hidden_return += info[INFO_HIDDEN_REWARD]

                            rgb_list.append(env.render("rgb_array"))
                            ansi_list.append(env.render("ansi"))
                            self._check_action_observation_valid(env, action, obs)
                            self._check_reward(env, reward)

                        self.assertEqual(done, demo.terminates)
                        self._check_rewards(
                            env,
                            cheat,
                            demo,
                            epsiode_info_observed_return,
                            episode_info_hidden_return,
                            episode_return,
                        )

                        self._check_rgb(rgb_list)
                        self._check_ansi(ansi_list)


if __name__ == "__main__":
    unittest.main()
