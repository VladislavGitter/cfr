import numpy as np
from random import shuffle
import time
import sys


class Kunh:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 4
        self.nash_equilibrium = dict()
        self.current_player = 0
        self.deck = np.array([0,1,2,3,0,1,2,3])
        self.n_actions = 2

    def train(self, n_iterations=50000):
        ev_0 = 0
        ev_1 = 0
        ev_2 = 0
        for _ in range(n_iterations):
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            # self.deck = np.array([0,1,3,2,1,2,3,0])
            
            log('[Kuhn.train]-deck: ' + str(self.deck))
            util = self.cfr('',1,1,1,'')
            ev_0 += util[0]
            ev_1 += util[1]
            ev_2 += util[2]
            # if _ % 1 == 0: 
            if _%50==0:print(str(_) + '[Kuhn.train]-//ev P0: ' + str(ev_0/(_+1)) + ' //ev P1: ' + str(ev_1/(_+1))+ ' //ev P2: ' +  str(ev_2/(_+1)))
            
            for _, v in self.nodeMap.items():
                v.update_strategy()
                # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
            log('[Kuhn.train]-ITERATION END')
            log('=======================================')
        # for _, v in self.nodeMap.items():
            # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
        ev_0 /= n_iterations
        display_results(ev_0, self.nodeMap)

    def cfr(self, history, pr_0, pr_1, pr_2, logStr):
        
        player = len(history) % 3
        plStr = 'P'+ str(player)
        player_card = self.deck[player]
        log(logStr + '[Kuhn.cfr]-START!!, history: ' + history + ' pr_0:' + str(pr_0) + ' pr_1:' + str(pr_1) + ' pr_2:' + str(pr_2) 
            + ' player:' +str(player)+ ' card:'+str(player_card))

        if self.is_terminal_3(history):
            rewards = self.showdown(history)
            log(logStr + '[Kuhn.cfr]-TERMINAL, reward: ' + str(rewards))
            return rewards

        node = self.get_node(player_card, history)
        strategy = node.strategy
        log(logStr + '[Kuhn.cfr]-node: ' + str(node.key) + ' regret_sum: ' + str(node.regret_sum) + ' strategy: ' + str(strategy))
        # Counterfactual utility per action.
        action_utils = np.zeros((self.n_actions, 3), dtype=float)

        log(logStr + '[Kuhn.cfr]-calculating counterfactual utils for ' + plStr + '(' + node.key  +')')
        
        for act in range(self.n_actions):
            next_history = history + node.action_dict[act]
            if player == 0:
                action_utils[act] = self.cfr(next_history, pr_0 * strategy[act], pr_1, pr_2, logStr + '    ')
            elif player == 1:
                action_utils[act] = self.cfr(next_history, pr_0, pr_1 * strategy[act], pr_2, logStr + '    ')
            elif player == 2:
                action_utils[act] = self.cfr(next_history, pr_0, pr_1, pr_2 * strategy[act], logStr + '    ')
        log(logStr + '[Kuhn.cfr]-calculated c. utils: ' + str(action_utils))
        # Utility of information set.
        action_utils_tr = np.transpose(action_utils)
        util = [sum(act_util * strategy) for act_util in action_utils_tr]
        util_player = sum(action_utils_tr[player] * strategy)

        log(logStr + '[Kuhn.cfr]-util(action_utils*strategy): ' + str(util))
        regrets = action_utils_tr[player] - util_player
        log(logStr + '[Kuhn.cfr]-regrets(action_utils-util): ' + str(regrets))
        log(logStr + '[Kuhn.cfr]-updating regret_sum, reach_pr...')
        if player == 0:
            node.reach_pr += pr_0
            node.regret_sum += pr_2 * regrets
        elif player == 1:
            node.reach_pr += pr_1
            node.regret_sum += pr_0 * regrets
        elif player == 2:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regrets
        log(logStr + '[Kuhn.cfr]-updated regret_sum, reach_pr: ' + str(node.regret_sum) + ', ' +  str(node.reach_pr))
        log(logStr + '[Kuhn.cfr]-END!!, history: ' + history + ' return:' + str(util))
        return util

    @staticmethod
    def is_terminal_3(history):
        # c0 special case 'ppbp', return false as it contradicts case 1 but its not terminal
        # c1 if 3p or 3b in history
        # c2 if first char is b and len 3
        # c3 if first two chars pb and len 4
        if history == 'ppbp':
            return False
        if history.count('b') >= 3 or history.count('p') >= 3:
            return True
        if len(history) == 3 and history[0] == 'b':
            return True
        if len(history) == 4 and history[0:2] == 'pb':
            return True
        return False
    def showdown(self,history):
        # return 1
        # count pot by history
        # detect folds by history
        winner_card = 0
        n_players = 3
        bets = [1 for _ in range(n_players)]
        folds = [0 for _ in range(n_players)]
        rewards = [0 for _ in range(n_players)]
        isBet = False #indicates if nex player has to call o can bet/check
        cards = list(self.deck[:n_players])
        for i in range(len(history)):
            if history[i] == 'b':
                isBet = True
                bets[i%3] += 3
            elif history[i] == 'p':
                if isBet:
                    folds[i%3] = 1
                    cards[i%3] = -1
        winner_card = max(cards)
        n_winners = cards.count(winner_card)
        pot = sum(bets)
        for i in range(n_players):
            if cards[i] == winner_card and folds[i] != 1:
                rewards[i] = pot/n_winners - bets[i]
            else:
                rewards[i] = -1 * bets[i]
        return rewards
    
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
    # print('player 1 expected value: {}'.format(ev))
    # print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 0 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0][x[0].index(' ')+1:]) % 3 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 1 strategies:')
    for _, v in filter(lambda x: len(x[0][x[0].index(' ')+1:]) % 3 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0][x[0].index(' ')+1:]) % 3 == 2, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))


if __name__ == "__main__":
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!                                 !')
    print('!        KUHN PLAYERS CRM         !')
    print('!                                 !')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=5000)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
