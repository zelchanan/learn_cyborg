from CybORG.CybORG import CybORG
from gym import Env

from CybORG.Agents import B_lineAgent, GreenAgent
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper, RedTableWrapper, EnumActionWrapper, FixedFlatWrapper

SCENARIO_PATH = "CybORG/Shared/Scenarios/Scenario1b.yaml"


class MyChallengeWrapper(OpenAIGymWrapper):
    def __init__(self, agent_name: str, env: BaseWrapper = None, max_counter: int = 60):
        super().__init__(agent_name=agent_name, env=env)
        self.step_counter: int = 0
        self.max_counter = max_counter
        self.previous_reward: int = 0

    def step(self, action=None):
        self.step_counter += 1
        obs, reward, done, info = super().step(action=action)

        done = done | self.step_counter >= self.max_counter
        #new_reward = reward - self.previous_reward
        self.previous_reward = reward
        return obs, reward, done, info

    def reset(self, agent=None):
        # np.random.seed(0)
        self.previous_reward = 0
        self.step_counter = 0
        return super().reset()


def get_env(path: str) -> MyChallengeWrapper:
    agents = {
        'Red': B_lineAgent,
        'Green': GreenAgent
    }

    env = CybORG(path, 'sim', agents=agents)
    eer = EnumActionWrapper(env)
    wrappers = FixedFlatWrapper(eer)
    # env = OpenAIGymWrapper(env=wrappers, agent_name='Blue')
    env = MyChallengeWrapper(env=wrappers, agent_name="Blue")

    # env = table_wrapper(env, output_mode='vector')
    # env = EnumActionWrapper(env)
    # env = OpenAIGymWrapper(agent_name=agent_name, env=env)
    # env = ChallengeWrapper(env=env, agent_name='Blue')
    return env
