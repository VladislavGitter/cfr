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
        self.action_dict = {}

    def train(self, n_iterations=50000):
        expected_game_value = 0
        for _ in range(n_iterations):
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            log('[Kuhn.train]-deck: ' + str(self.deck))
            expected_game_value += self.cfr('', 1, 1, '')
            if _ % 10 == 0: print(str(_) + '[Kuhn.train]-expected_game_value: ' + str(expected_game_value/_))
            for _, v in self.nodeMap.items():
                # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
                v.update_strategy()
            log('[Kuhn.train]-ITERATION END')
            log('=======================================')
        for _, v in self.nodeMap.items():
            log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + ')' + str(v.action_dict) + ': ' + str(v.strategy))
        expected_game_value /= n_iterations
        display_results(expected_game_value, self.nodeMap)

    def cfr(self, history, pr_1, pr_2, logStr):
        log(logStr + '[Kuhn.cfr]-START!!, history: ' + history + ' pr_1:' + str(pr_1) + ' pr_2:' + str(pr_2))
        
        self.action_dict = self.get_action_dict(history)
        n_actions = len(self.action_dict)
        in_flop = 'F' in history
        flop_history = history[history.rfind(' ') +1:] if in_flop else history
        n = len(flop_history)
        is_player_1 = n % 2 == 0
        plStr = 'P1' if is_player_1 else 'P2'
        log(logStr + '[Kuhn.cfr]-player1: ' + str(is_player_1))
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        log(logStr + '[Kuhn.cfr]-player_card: ' + str(player_card))
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            reward = self.get_reward( history, card_player, card_opponent, self.deck[2:3])
            log(logStr + '[Kuhn.cfr]-TERMINAL, history: ' + history)
            log(logStr + '[Kuhn.cfr]-TERMINAL, reward: ' + str(reward))
            return reward
        
        if self.start_flop(history):
            is_player_1 == True
            player_card = self.deck[0]
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            plStr = 'P1'
            pot = self.get_pot(history)
            flopStr = str(pot) + 'F' + str(self.deck[2]) + ' '
            history += flopStr

        node = self.get_node(player_card, history)
        strategy = node.strategy

        log(logStr + '[Kuhn.cfr]-node: ' + str(node.key) + ' regret_sum: ' + str(node.regret_sum) + ' strategy: ' + str(strategy))
        # Counterfactual utility per action.
        action_utils = np.zeros(n_actions)
        log(logStr + '[Kuhn.cfr]-calculating counterfactual utils for ' + plStr + '(' + node.key  +')')
        
        for act in range(n_actions):
            next_history = history + node.action_dict[act]
            if is_player_1:
                action_utils[act] = -1 * self.cfr(next_history, pr_1 * strategy[act], pr_2, logStr + '    ')
            else:
                action_utils[act] = -1 * self.cfr(next_history, pr_1, pr_2 * strategy[act], logStr + '    ')
        log(logStr + '[Kuhn.cfr]-calculated c. utils for' +plStr+ ': ' + str(action_utils))
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
    def get_action_dict(history):
        history = history[history.rfind(' ') +1:] if 'F' in history else history
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
    def is_terminal(history):
        in_flop = 'F' in history
        if history[-2:] == 'bp':
            return True
        elif in_flop and (history[-2:] == 'pp' or history[-2:] == "bc"):
            return True
    
    @staticmethod
    def start_flop(history):
        return ('F' not in history and (history[-2:] == 'pp' or history[-2:] == "bc"))
    def get_reward(self, history, player_card, opponent_card, board_cards):
        pot = self.get_pot(history)
        terminal_pass = history[-1] == 'p'
        bet_call = history[-2:] == "bc"
        in_flop = 'F' in history
        winner = self.get_winner_showdown(player_card, opponent_card,board_cards)
        if terminal_pass:
            if history[-2:] == 'pp' and in_flop:
                if winner == 1: return pot/2
                elif winner == -1: return -pot/2
                else: return 0
            else: #bet_pass
                # subtract 2 from pot as the other player doesnt call, so the last bet/raise doesnt count for reward
                return (pot-2)/2
        elif bet_call and in_flop:
            if winner == 1: return pot/2
            elif winner == -1: return -pot/2
            else: return 0
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
        pot = 2 # 2 blinds, 1bb each
        to_call = 0
        for action in history:
            if action == 'b':
               pot += 2 + to_call
               to_call = 2
            elif action == 'c':
                pot += to_call
                to_call = 0
        # print('h:' + history + ' pot: ' + str(pot))
        return pot

    def get_node(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
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

def log(message):
    if len(message) < 200:
        if False: print(message)

def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
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
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=20)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
