import copy
import inspect, pprint
from typing import Union

import pandas as pd

from CybORG.Agents.SimpleAgents import BaseAgent
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Shared import Results


class EnumActionWrapper(BaseWrapper):
    def __init__(self, env: Union[type, BaseWrapper] = None, agent: BaseAgent = None):
        super().__init__(env, agent)
        self.possible_actions = None
        self.reverse_possible_actions = None
        self.action_signature = {}
        #self.current_agent = "Red"
        self.get_action_space('Red')

    def step(self, agent=None, action: int = None) -> Results:
        #print(agent)
        if action is not None:
            #print(self.reverse_possible_actions[action])
            action = self.possible_actions[action]
        return super().step(agent, action)

    def reverse_action_space(self):
        #current_agent = "unknown" if self.current_agent is None else self.current_agent
        agent = vars(self.possible_actions[-1])["agent"]
        self.reverse_possible_actions = agent+" "+pd.Series(self.possible_actions).astype(str)

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))

        self.possible_actions = possible_actions
        self.reverse_action_space()
        return len(possible_actions)
