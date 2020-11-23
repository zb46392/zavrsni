from . import ObservableEnvironment as O_Env
from agent.base import Base
from random import random, randint
from typing import Tuple, Any, Optional
from datetime import datetime


class QLearning(Base):
    def __init__(self, training_amount: int = 100000):
        super(Base, self).__init__()
        self._q_table = {}
        self._previous_action = None
        self._current_state = None
        self._previous_state = None
        self._init_q_value = 0.0
        self._learning_rate = 0.3
        self._gamma = 0.9
        self._epsilon = 1.0
        self._epsilon_decay = self._epsilon / training_amount

    def prepare_for_episode(self) -> None:
        self._previous_action = None
        self._previous_state = None

    def observe_environment(self, env: O_Env) -> None:
        if env.state not in self._q_table.keys():
            self._add_state_to_q_table(env.state)

        self._update_q_table(env)
        self._current_state = env.state

        if env.is_terminal and self._epsilon > 0:
            self._epsilon -= self._epsilon_decay

    def choose_action(self, actions: Tuple[Any, ...]) -> Optional[Any]:
        for action in actions:
            if action not in self._q_table[self._current_state].keys():
                self._add_action_to_q_table(action)

        if random() < self._epsilon:
            action = self._choose_random_action(actions)
        else:
            action = self._choose_best_action(actions)

        self._previous_state = self._current_state
        self._previous_action = action

        return action

    def _add_state_to_q_table(self, state: Tuple[int, ...]) -> None:
        self._q_table[state] = {}

    def _add_action_to_q_table(self, action: Any) -> None:
        self._q_table[self._current_state][action] = self._init_q_value

    def _update_q_table(self, env: O_Env) -> None:
        # Calculate new Q-Value (Bellman equation)
        if len(self._q_table[env.state].keys()) > 0 \
                and self._previous_state is not None \
                and self._previous_action is not None:
            current_action = self._get_actions_by_value(env.state)[0]
            old_q = self._q_table[
                self._previous_state][self._previous_action]

            new_q = (1 - self._learning_rate) * old_q + \
                    self._learning_rate * (env.reward + self._gamma * 
                                            self._q_table[env.state][current_action])

            self._q_table[self._previous_state][self._previous_action] = new_q
        elif env.is_terminal:
            self._q_table[self._previous_state][self._previous_action] = \
                env.reward

    @staticmethod
    def _choose_random_action(actions: Tuple[Any, ...]) -> Any:
        rnd = randint(0, len(actions) - 1)

        return actions[rnd]

    def _choose_best_action(self, actions: Tuple[Any, ...]) -> Any:
        actions_by_value = self._get_actions_by_value(self._current_state)
        h_value = self._q_table[self._current_state][actions_by_value[0]]
        h_actions = []
        for action in actions_by_value:
            if self._q_table[self._current_state][action] == h_value \
                    and action in actions:
                h_actions.append(action)

        if len(h_actions) > 0:
            rnd = randint(0, len(h_actions) - 1)
            return h_actions[rnd]

        return None

    def _get_actions_by_value(self, state: Tuple[int, ...]) -> \
            Tuple[Any, ...]:
        return tuple(
            sorted(
                self._q_table[state].keys(), 
                key=lambda action: self._q_table[state][action],
                reverse=True)
        )

