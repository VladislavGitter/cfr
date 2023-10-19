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
        self.deck = np.array([0,1,2])
        self.n_actions = 2

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            print('[Kuhn.train]-deck: ' + str(self.deck))
            util = self.cfr('', 1, 1,'')
            expected_game_value += util
            print('[Kuhn.train]-expected_game_value: ' + str(expected_game_value/_))
            print(str(util))
            for _, v in self.nodeMap.items():
                v.update_strategy()
                log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
            log('[Kuhn.train]-ITERATION END')
            log('=======================================')
        for _, v in self.nodeMap.items():
            log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def cfr(self, history, pr_1, pr_2, logStr):
        log(logStr + '[Kuhn.cfr]-START!!, history: ' + history + ' pr_1:' + str(pr_1) + ' pr_2:' + str(pr_2))
        n = len(history)
        is_player_1 = n % 2 == 0
        plStr = 'P1' if is_player_1 else 'P2'
        log(logStr + '[Kuhn.cfr]-player1: ' + str(is_player_1))
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        log(logStr + '[Kuhn.cfr]-player_card: ' + str(player_card))
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            reward = self.get_reward(history, card_player, card_opponent)
            log(logStr + '[Kuhn.cfr]-TERMINAL, reward: ' + str(reward))
            return reward

        node = self.get_node(player_card, history)
        strategy = node.strategy
        log(logStr + '[Kuhn.cfr]-node: ' + str(node.key) + ' regret_sum: ' + str(node.regret_sum) + ' strategy: ' + str(strategy))
        # Counterfactual utility per action.
        action_utils = np.zeros(self.n_actions)

        log(logStr + '[Kuhn.cfr]-calculating counterfactual utils for ' + plStr + '(' + node.key  +')')
        
        for act in range(self.n_actions):
            next_history = history + node.action_dict[act]
            if is_player_1:
                action_utils[act] = -1 * self.cfr(next_history, pr_1 * strategy[act], pr_2, logStr + '    ')
            else:
                action_utils[act] = -1 * self.cfr(next_history, pr_1, pr_2 * strategy[act], logStr + '    ')
        log(logStr + '[Kuhn.cfr]-calculated c. utils: ' + str(action_utils))
        # Utility of information set.
        util = sum(action_utils * strategy)
        log(logStr + '[Kuhn.cfr]-util(action_utils*strategy): ' + str(util))
        regrets = action_utils - util
        log(logStr + '[Kuhn.cfr]-regrets(action_utils-util): ' + str(regrets))
        log(logStr + '[Kuhn.cfr]-updating regret_sum, reach_pr...')
        if is_player_1:
            node.reach_pr += pr_1
            node.regret_sum += pr_2 * regrets
        else:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regrets
        log(logStr + '[Kuhn.cfr]-updated regret_sum, reach_pr: ' + str(node.regret_sum) + ', ' +  str(node.reach_pr))
        log(logStr + '[Kuhn.cfr]-END!!, history: ' + history + ' return:' + str(util))
        return util

    @staticmethod
    def is_terminal(history):
        if history[-2:] == 'pp' or history[-2:] == "bb" or history[-2:] == 'bp':
            return True

    @staticmethod
    def get_reward(history, player_card, opponent_card):
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == "bb"
        if terminal_pass:
            if history[-2:] == 'pp':
                return 1 if player_card > opponent_card else -1
            else:
                return 1
        elif double_bet:
            return 2 if player_card > opponent_card else -2

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            action_dict = {0: 'p', 1: 'b'}
            info_set = Node(key, action_dict)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


class Node:
    def __init__(self, key, action_dict, n_actions=2):
        self.key = key
        self.n_actions = n_actions
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

def log(message):
    if False: print(message)

def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))


if __name__ == "__main__":
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!                                 !')
    print('!             KUHN CRM            !')
    print('!                                 !')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=2000)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
