import numpy as np
from random import shuffle
import time
import sys


class Kunh:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 3
        self.nash_equilibrium = dict()
        self.current_player = 0
        self.deck = np.array([0, 0, 1, 1, 2, 2])
        # self.n_actions = 2
        self.action_dict = {}

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            shuffle(self.deck)
            expected_game_value += self.cfr('', 1, 1)
            for _, v in self.nodeMap.items():
                v.update_strategy()
        for _, v in self.nodeMap.items():
            print('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + ')' + str(v.action_dict) + ': ' + str(v.strategy))
        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def cfr(self, history, pr_1, pr_2):
        # action_dict = {0: 'p', 1: 'b'}
        self.action_dict = self.get_action_dict(history)

        n_actions = len(self.action_dict)

        n = len(history)
        is_player_1 = n % 2 == 0
        player_card = self.deck[0] if is_player_1 else self.deck[1]

        if self.is_terminal_preflop(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            reward = self.get_reward_preflop( history, card_player, card_opponent)
            return reward

        node = self.get_node(player_card, history)
        strategy = node.strategy

        # Counterfactual utility per action.
        action_utils = np.zeros(n_actions)

        for act in range(n_actions):
            next_history = history + node.action_dict[act]
            if is_player_1:
                action_utils[act] = -1 * self.cfr(next_history, pr_1 * strategy[act], pr_2)
            else:
                action_utils[act] = -1 * self.cfr(next_history, pr_1, pr_2 * strategy[act])

        # Utility of information set.
        util = sum(action_utils * strategy)
        regrets = action_utils - util
        if is_player_1:
            node.reach_pr += pr_1
            node.regret_sum += pr_2 * regrets
        else:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regrets

        return util
    
    @staticmethod
    def get_action_dict(history):
        action_dict = {0: 'p', 1: 'b'}
        b_counter = history.count('b')
        if b_counter == 0:
            action_dict =  {0: 'p', 1: 'b'}
        elif b_counter == 1:
            action_dict =  {0: 'p', 1: 'b', 2:'c'}
        elif b_counter == 2:
            action_dict =  {0: 'p', 1:'c'}
        return action_dict

    @staticmethod
    def is_terminal_preflop(history):
        # if history[-2:] == 'pp' or history[-2:] == "bc" or history[-2:] == 'bp':
        #     return True
        if history[-2:] == 'pp' or history[-2:] == 'bp':
            return True

    def get_reward_preflop(self, history, player_card, opponent_card):
        pot = self.get_pot(history)
        if player_card > opponent_card:
            return pot
        elif player_card == opponent_card:
            return pot/2
        elif player_card < opponent_card:
            return -pot
        
    def get_pot(self,history):
        pot = 2 # 2 blinds 1bb each
        to_call = 0
        # ejemplo 3 pbbc
        history_no_card = history[2:]
        for element in range(0, len(history_no_card)):
            if element == 'b':
               pot += 2 + to_call
               to_call += 2
            elif element == 'c':
                pot += to_call
        return pot

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            # action_dict = {0: 'p', 1: 'b'}
            info_set = Node(key, self.action_dict)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


class Node:
    def __init__(self, key, action_dict):
        self.key = key
        self.n_actions = len(action_dict)
        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)
        self.action_dict = action_dict
        self.strategy = np.repeat(1/self.n_actions, self.n_actions)
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def update_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.reach_pr_sum += self.reach_pr
        self.strategy = self.get_strategy()
        self.reach_pr = 0

    def get_strategy(self):
        regrets = self.regret_sum
        regrets[regrets < 0] = 0
        normalizing_sum = sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            return np.repeat(1/self.n_actions, self.n_actions)

    def get_average_strategy(self):
        strategy = self.strategy_sum / self.reach_pr_sum
        # Re-normalize
        total = sum(strategy)
        strategy /= total
        return strategy

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):
        print(v)
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):
        print(v)


if __name__ == "__main__":
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=5000000)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
