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
        self.deck = np.array([0,0,1,1,2,2])
        self.n_actions = 2

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            log('[Kuhn.train]-deck: ' + str(self.deck))
            expected_game_value += self.cfr('', 1, 1,'')
            if _ % 100 == 0:print(str(_) + '[Kuhn.train]-expected_game_value: ' + str(expected_game_value/_))
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
        flop_history = history[history.rfind(' ') +1:]
        n = len(flop_history)
        is_player_1 = n % 2 == 0
        plStr = 'P1' if is_player_1 else 'P2'
        log(logStr + '[Kuhn.cfr]-player1: ' + str(is_player_1))
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        log(logStr + '[Kuhn.cfr]-player_card: ' + str(player_card))

        
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            reward = self.get_reward(history, card_player, card_opponent, self.deck[2:3])
            log(logStr + '[Kuhn.cfr]-TERMINAL, reward: ' + str(reward))
            return reward

        if self.start_flop(history):
            is_player_1 == True
            player_card = self.deck[0]
            self.action_dict = {0: 'p', 1: 'b'}
            plStr = 'P1'
            pot = self.get_pot(history)
            flopStr = ' ' + str(pot) + 'F' + str(self.deck[2]) + ' '
            history += flopStr

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
        in_flop = 'F' in history
        if history[-2:] == 'bp':
            return True
        elif in_flop and (history[-2:] == 'pp' or history[-2:] == "bb"):
            return True
    
    @staticmethod
    def start_flop(history):
        in_flop = 'F' in history
        if not in_flop and (history[-2:] == 'pp' or history[-2:] == "bb"):
            return True
        return False

    def get_reward(self, history, player_card, opponent_card, board_cards):
        pot = self.get_pot(history)
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == "bb"
        in_flop = 'F' in history
        winner = self.get_winner_showdown(player_card, opponent_card,board_cards)
        if terminal_pass:
            if history[-2:] == 'pp' and in_flop:
                if winner == 1: return pot/2
                elif winner == -1: return -pot/2
                else: return pot/4
            else:
                return pot/2
        elif double_bet and in_flop:
            if winner == 1: return pot/2
            elif winner == -1: return -pot/2
            else: return pot/4


    @staticmethod    
    def get_winner_showdown(player_card, opponent_card, board_cards):
        # 1 if player wins, 0 draw, -1 if opponent wins
        #leduc version, checks pairs for every player, if neither, checks high card
        if player_card == board_cards[0]:
            return 1
        elif opponent_card == board_cards[0]:
            return -1
        elif player_card > opponent_card:
            return 1
        elif player_card < opponent_card:
            return -1
        return 0
    @staticmethod
    def get_pot(history):
        pot = 2 # 2 blinds 1bb each
        if 'F' in history:
            pot = int(history[history.index('F')-1])
            flopHistory = history[history.rfind(' ')+1:]
            for act in flopHistory:
                if act == 'b': pot += 2
        else:
            for act in history:
                if act == 'b': pot += 1
        return pot
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
    if True: print(message)

def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    # sorted_preflop_items = filter(lambda x: 'F' not in x[0], sorted_items)
    # sorted_preflop_items = []
    # sorted_flop_items = []
    # for item in sorted_items:
    #     if 'F' in item[0]:
    #         sorted_flop_items.append(item)
    #     else:
    #         sorted_preflop_items.append(item)
    # print(str(sorted_preflop_items))
    print()
    print('player 1 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 0 and 'F' not in x[0], sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1 and 'F' not in x[0], sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    

    print('player 1 FLOP strategies:')
    for _, v in filter(lambda x: 'F' in x[0] and len(x[0][x[0].index('F') + 2:]) % 2 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 FLOP strategies:')
    for _, v in filter(lambda x: 'F' in x[0] and len(x[0][x[0].index('F') + 2:]) % 2 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))


if __name__ == "__main__":
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!                                 !')
    print('!             KUHN CRM            !')
    print('!                                 !')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=200)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
