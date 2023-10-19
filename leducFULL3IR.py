import copy
import numpy as np
from random import shuffle
import time
import sys

from datetime import datetime



class Kunh:

    def __init__(self):
        self.nodeMap = {}
        self.expected_game_value = 0
        self.n_cards = 6
        self.nash_equilibrium = dict()
        self.current_player = 0
        self.deck = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
        self.action_dict = {}
        # self.traverser = 0

        self.rewardsMap = {}

    def train(self, n_iterations=50000):
        ev_0 = 0
        ev_1 = 0
        ev_2 = 0
        for _ in range(n_iterations):
            time1 = time.time()
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            # print(self.deck)
            ev_l = [ev_0,ev_1,ev_2]
            # self.traverser = ev_l.index(min(ev_l))
            # self.deck = np.array([0,1,3,2,1,2,3,0])
            
            log('[Kuhn.train]-deck: ' + str(self.deck))
            initial_config = {
                'act_pls':[1,1,1],
                'bets':[0.5,1,0]}
            util = self.cfr('',1,1,1,initial_config,'')
            ev_0 += util[0]
            ev_1 += util[1]
            ev_2 += util[2]
            
            # if _ % 10 == 0: print(str(_) + '[Kuhn.train]-expected_game_value: ' + str(ev_0/_))
            myInt = _ * 1
            num = 2
            # if _<n_iterations * 2 / 3:
            #     num = 50
            # if (_-1)%num==0: 
            for _, v in self.nodeMap.items():
                v.update_strategy()
                # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
            log('[Kuhn.train]-ITERATION END')
            log('=======================================')
            if myInt%1==0:
                print(str(myInt) + '[Kuhn.train]-//ev P0:' + str(ev_0/(myInt+1))+' //ev P1:'+str(ev_1/(myInt+1))+' //ev P2:'+str(ev_2/(myInt+1))+' //t:'+ str(abs(time1 - time.time())))
        # for _, v in self.nodeMap.items():
            # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
        ev_0 /= n_iterations
        display_results(ev_0, self.nodeMap)
        save_results(self.nodeMap)

    def cfr(self, history, pr_0, pr_1, pr_2,config, logStr):
        self.action_dict = self.get_action_dict(history)
        n_actions = len(self.action_dict)
        player = self.get_active_player(history, config)

        # log(logStr + '[Kuhn.cfr]-START!!, history: ' + history + ' pr_0:' + str(pr_0) + ' pr_1:' + str(pr_1) + ' pr_2:' + str(pr_2) 
        #     + ' player:' +str(player)+ ' card:'+str(self.deck[player]))
        if self.is_terminal(history, config):
            rewards = self.showdown(history,config)
            log(logStr + 'NODE, terminal: h: ' + history + ' rewards: ' + str(rewards))
            return rewards
        bd_cards = []
        if self.start_flop(history):
            player = config['act_pls'].index(1)
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            # bets = self.get_bets_preflop(history)
            pot = sum(config['bets'])
            # config['bets'] = bets
            plFlop = ''
            for i in config['act_pls']:
                plFlop += str(i)
            # flopStr = ' ' + str(pot) + 'F' + plFlop + ' '

            flopStr =  str(pot) + 'F' + plFlop + ' '
            history = flopStr
            log(logStr + 'NODE FLOP: h: ' + history + ' n_pl: ' + str(player))
        
        elif self.start_turn(history):
            player = config['act_pls'].index(1)
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            # bets = self.get_bets_preflop(history)
            pot = sum(config['bets'])
            # config['bets'] = bets
            plFlop = ''
            for i in config['act_pls']:
                plFlop += str(i)
            # flopStr = ' ' + str(pot) + 'T' + plFlop + ' '
            flopStr = str(pot) + 'T' + plFlop + ' '
            history = flopStr
            log(logStr + 'NODE TURN: h: ' + history + ' n_pl: ' + str(player))

        elif self.start_river(history):
            player = config['act_pls'].index(1)
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            # bets = self.get_bets_preflop(history)
            pot = sum(config['bets'])
            # config['bets'] = bets
            plFlop = ''
            for i in config['act_pls']:
                plFlop += str(i)
            # flopStr = ' ' + str(pot) + 'R' + plFlop + ' '
            flopStr = str(pot) + 'R' + plFlop + ' '
            history = flopStr
            log(logStr + 'NODE RIVER: h: ' + history + ' n_pl: ' + str(player))
        else:
            log(logStr + 'NODE DECISION: h: ' + history + ' pl: ' + str(player))
        state = self.get_state(history)
        if state != 'P':
            bd_cards = self.get_bd_cards_state(history)
        node = self.get_node(self.deck[player], bd_cards, history, player)
        strategy = node.strategy

        action_utils = np.zeros((n_actions, 3), dtype=float)
        # log(logStr + '[Kuhn.cfr]-calculating counterfactual utils for ' + plStr + '(' + node.key  +')')
        
        for act in range(n_actions):
            next_history = history + node.action_dict[act]
            next_config = copy.copy(config)
            # next_config['act_pls'] = self.update_pl_config(next_history, config['act_pls'])
            
            next_config['act_pls'] = self.update_pl_config(next_history, config['act_pls'])
            next_config['bets'] = self.update_bets_config(player, next_history, config['bets'])
            # log(logStr + 'bets for ' + history + ',' + str(next_config['bets']) + ' act_pls:' + str( next_config['act_pls']))
            # print('update_config h:' + next_history + ' ' + str(next_config))
            if player == 0:
                action_utils[act] = self.cfr(next_history, pr_0 * strategy[act], pr_1, pr_2,next_config, logStr + '    ')
            elif player == 1:
                action_utils[act] = self.cfr(next_history, pr_0, pr_1 * strategy[act], pr_2,next_config, logStr + '    ')
            elif player == 2:
                action_utils[act] = self.cfr(next_history, pr_0, pr_1, pr_2 * strategy[act],next_config, logStr + '    ')
        action_utils_tr = np.transpose(action_utils)
        util = [sum(act_util * strategy) for act_util in action_utils_tr]
        util_player = sum(action_utils_tr[player] * strategy)
        regrets = action_utils_tr[player] - util_player
        if player == 0:
            node.reach_pr += pr_0
            node.regret_sum += pr_2 * regrets
        elif player == 1:
            node.reach_pr += pr_1
            node.regret_sum += pr_0 * regrets
        elif player == 2:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regrets
        # log(logStr + '[Kuhn.cfr]-END!!, history: ' + history + ' return:' + str(util))
        return util
    
    def get_bd_cards_state(self, history):
        board_cards = []
        if 'R' in history:
            board_cards.append(self.deck[2])
            board_cards.append(self.deck[3])
            board_cards.append(self.deck[4])
            return board_cards
        if 'T' in history:
            board_cards.append(self.deck[2])
            board_cards.append(self.deck[3])
            return board_cards
        if 'F' in history:
            board_cards.append(self.deck[2])
            return board_cards
        return []

    def get_active_player(self, history, config):
        player = -1

        state = self.get_state(history)
        # in_flop = 'F' in history
        
        active_history = history[history.rfind(' ') +1:] if state != 'P' else history
        # n_active_players = config['act_pls'].count(1)
        n_active_players = config['act_pls'].count(1)
        if state != 'P':
            flop_type = history[history.index(state)+1:history.rfind(' ')].count('1')
            if flop_type == 3:
                player = len(active_history) % 3
            else:
                abs_pl = len(active_history) % n_active_players
                player = [i for i, n in enumerate(config['act_pls']) if n == 1][abs_pl]
        else:
            if len(history) == 3 and history[0] == 'p' :
                player = 0
            else:
                player = (len(active_history) + 2) % 3 
        return player
    
    @staticmethod
    def update_bets_config(player,next_history, bets):
        result = copy.copy(bets)
        if next_history[-1] == 'b':
            result[player] = max(bets) * 2
        if next_history[-1] == 'c':
            result[player] = max(bets)
        return result

    def update_pl_config(self, history, act_pls):
        # very abstract function that updates a list of of ints which represents if fold/not fold state for every player,
        # helps to find the active player for next recursion step, NOT SCALABLE TO MORE PLAYERS CASE NEITHER FOR 2 PLAYER CASE
        # tested it manually for every possible case, GL touching it ;)
        result = copy.copy(act_pls)
        state = self.get_state(history)
        act_history =  history[history.rfind(' ') +1:] if 'F' in history else history
        if len(act_history) == 0:
            return result

        if state == 'P':
            if len(history) <=2 and history[-1] == 'p':
                act_pl = (len(history) + 1)%3
                result[act_pl] = 0
            if len(history) == 3 and history[-1] == 'p' and history.count('b') != 0:
                act_pl = 1
                result[act_pl] = 0
            if len(history) == 4:
                pl_act = 0 if history[0] == 'p' else 2
                if history[-1] == 'p' and history.count('b') != 0:
                    result[pl_act] = 0
            if len(history) == 5:
                pl_act = 0
                if history[-1] == 'p' and history[-3:].count('b') != 0:
                    result[pl_act] = 0
        else:
            f_history =  history[history.rfind(' ') +1:]
            n_act_pls = history[history.index(state)+1:history.rfind(' ')].count('1')
            if n_act_pls == 3:
                act_pl = (len(f_history)-1)%3
                if len(f_history) != 1:
                    if f_history[-1] == 'p' and f_history[-3:].count('b') > 0:
                        result[act_pl] = 0
            if n_act_pls == 2:
                act_pl_ind = [act_pls.index(i) for i in act_pls if i == 1]
                if len(f_history) >1:
                    act_pl = (len(f_history)-1)%2
                    act_pl_ind = []
                    for i in range(len(act_pls)):
                        if act_pls[i] == 1:
                            act_pl_ind.append(i)
                    if f_history[-2:] == 'bp':
                        result[act_pl_ind[act_pl]] = 0
        return result

    def get_state(self, history):
        in_flop = 'F' in history
        in_turn = 'T' in history
        in_river = 'R' in history
        return 'R' if in_river else ('T' if in_turn else ('F' if in_flop else 'P'))

    def showdown(self,history, config):
        # rewards = self.get_saved_rewards(history)
        # if rewards != None: 
        #     # print('returning rewards')
        #     return rewards
        # else:
        #     rewards = self.calculate_showdown(config)
        #     bd_cards = self.get_bd_cards_state(history)
        #     in_game_cards = list(self.deck[0:3])
        #     in_game_cards.extend(bd_cards)
        #     key = str(in_game_cards) + history
        #     self.rewardsMap[key] = copy.copy(rewards)
        #     # print('saveing reward:' + key + ' ' + str(rewards))
        # return rewards
        return self.calculate_showdown(config)
    
    def calculate_showdown(self, config):
        rewards = [0,0,0]
        pot = sum(config['bets'])
        # queda un jugador
        if config['act_pls'].count(1) == 1:
            
            pl_win = config['act_pls'].index(1)
            for i in range(len(rewards)):
                rewards[i] = config['bets'][i] * -1 if i != pl_win else pot-config['bets'][i]
        else:
            winner_score = [-1,-1,-1,-1]
            scores = [[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]]
            for i in range(len(config['act_pls'])):
                if config['act_pls'][i] == 1:
                    pl_score = self.get_hand_score(self.deck[i], self.deck[3:6])
                    scores[i] = pl_score
                    for i in range(len(winner_score)):
                        if winner_score[i] < pl_score[i]:
                            winner_score = pl_score
                            break
            n_winners = scores.count(winner_score)
            for i in range(len(config['act_pls'])):
                if config['act_pls'][i] == 1:
                    rewards[i] = (config['bets'][i] * -1) if scores[i] != winner_score else ((pot/n_winners) - config['bets'][i])
                else:
                    rewards[i] = config['bets'][i] * -1
        return rewards

    def get_saved_rewards(self, history):
        bd_cards = self.get_bd_cards_state(history)
        in_game_cards = list(self.deck[0:3])
        in_game_cards.extend(bd_cards)
        key = str(in_game_cards) + history
        if key not in self.rewardsMap:
            return None
        return self.rewardsMap[key]

    # @staticmethod
    # def hand_score(pc,bc):
    #     score = [-1,-1]
    #     if pc == bc:
    #         score[0] = pc
    #     else:
    #         score[1] = max(pc,bc)
    #     return score
    def get_hand_score(self,pl_card, board_cards):
        # score = [set(-1,0,1,2), double(21,20,10), pair(2,1,0), high(2*3*5 etc)]
        score = [-1,-1,-1,-1]
        cardsCount = [0,0,0,0]
        cardsCount[pl_card] +=1
        for c in board_cards:
            cardsCount[c] +=1
        if any(x == 3 for x in cardsCount):
            score[0] = cardsCount.index(3)
            score[3] = self.get_prime_high(cardsCount)
        pairs = cardsCount.count(2)
        if pairs == 2:
            small_pair = cardsCount.index(2)
            big_pair = cardsCount.index(2) + 1
            
            score[1] = big_pair * 10 + small_pair
        elif pairs == 1:
            score[2] = cardsCount.index(2) 
            score[3] = self.get_prime_high(cardsCount)
        elif pairs == 0:
            score[3] = self.get_prime_high(cardsCount)
        return score
    

    @staticmethod
    def get_prime_high(cardsCount):
        result = 1
        for i in range(len(cardsCount)):
            if cardsCount[i] == 1:
                if i == 0:
                    result = result * 2
                if i == 1:
                    result = result * 3
                if i == 2:
                    result = result * 5
                if i == 3:
                    result = result * 7
        return result
        
    def get_action_dict(self, history):
        state = self.get_state(history)
        action_dict = {0: 'p', 1: 'b'}
        if state != 'P' and history[history.rfind(state) +1:history.rfind(' ')].count('1') == 3:
            history = history[history.rfind(' ') +1:]
            b_counter = history.count('b')
            # print(history)
            if len(history) == 0:
               action_dict =  {0: 'p', 1: 'b'}
            elif len(history) < 3:
                if b_counter == 0:
                    action_dict =  {0: 'p', 1: 'b'}
                else:
                    action_dict =  {0: 'p', 1: 'b', 2:'c'}
            elif len(history) == 3:
                if history[0] == 'b':
                    action_dict =  {0: 'p', 1:'c'}
                else:
                    action_dict =  {0: 'p', 1: 'b', 2:'c'}
            else:
                action_dict =  {0: 'p', 1:'c'}
        elif state != 'P' and history[history.rfind(state) +1:history.rfind(' ')].count('1') == 2:
            f_history = history[history.rfind(' ') +1:]
            b_counter = history.count('b')
            action_dict = {0: 'p', 1: 'b'}
            b_counter = f_history.count('b')
            if b_counter == 0:
                action_dict =  {0: 'p', 1: 'b'}
            elif b_counter == 1:
                action_dict =  {0: 'p', 1: 'b', 2:'c'}
            elif b_counter == 2:
                action_dict =  {0: 'p', 1:'c'}
            return action_dict
        elif state == 'P':
            b_counter = history.count('b')
            if len(history) < 3:
                action_dict =  {0: 'p', 1: 'b', 2:'c'}
                if len(history) == 2 and b_counter == 0:
                    action_dict =  {0: 'p', 1: 'b'}
            else:
                action_dict =  {0: 'p', 1:'c'}
        return action_dict
    
    def is_terminal(self, history, config):
        in_flop = 'F' in history
        in_river = 'R' in history
        state = self.get_state(history)
        if state == 'P':
            if history == 'pp':return True
            if len(history)>2 and history[0] == 'p' and history[-2:] == 'bp':return True
            if history[-3:] == 'bpp':return True
            if history[-4:] == 'bpbp':return True
            if history[-4:] == 'cpbp':return True
        elif state == 'R' and history[history.rfind('R') +1:history.rfind(' ')].count('1') == 3:
            if history[-3:] == 'ppp':return True
            if history[-3:] == 'bpp':return True
            if history[-3:] == 'bpp':return True
            if history[-3:] == 'bpc':return True
            if history[-3:] == 'bcp':return True
            if history[-3:] == 'bcc':return True
            if history[-4:] == 'bpbp':return True
            if history[-4:] == 'bpbc':return True
        elif state == 'R' and history[history.rfind('R') +1:history.rfind(' ')].count('1') == 2:
            if history[-2:] == 'bp':return True
            if history[-2:] == 'pp':return True
            if history[-2:] == "bc":return True
        elif config['act_pls'].count(1) == 1:
            return True
        return False
    
    def start_flop(self, history):
        state = self.get_state(history)
        if state == 'P':
            if history[-3:] == 'bcc' or history == 'ccp':return True
            if len(history)>2 and history[0] == 'p' and history[-2:] == 'bc':return True
            if history [-3:] == 'bpc' or history [-3:] == 'bcp':return True
            if history == 'pcp' or history == 'cpbp' or history == 'cpp':return True
            if history[-3:] == 'pbc':return True
        return 0
    
    def start_turn(self, history):
        state = self.get_state(history)
        if state == 'F' and len(history[history.rfind(' ') + 1:]) > 2:
            if history[-3:] == 'ppp' or history[-3:] == 'bcc' or history[-3:] == 'bcp' or history[-3:] == 'bpc':return True
            if history[-4:] == 'bpbc':return True
        return False
    
    def start_river(self, history):
        state = self.get_state(history)
        if state == 'T' and len(history[history.rfind(' ') + 1:]) > 2:
            if history[-3:] == 'ppp' or history[-3:] == 'bcc' or history[-3:] == 'bcp' or history[-3:] == 'bpc':return True
            if history[-4:] == 'bpbc':return True
        return False

    def get_node(self, card, bd_cards, history, player):
        cardsStr = '(' +  str(card) + ','
        for c in bd_cards:
            cardsStr = cardsStr + str(c) +','
        cardsStr = cardsStr.removesuffix(',') + ')'
        key = str(player)+cardsStr + history
        if key not in self.nodeMap:
            info_set = Node(key, self.action_dict, player)
            self.nodeMap[key] = info_set
            return info_set
        return self.nodeMap[key]


class Node:
    def __init__(self, key, action_dict, player):
        self.key = key
        self.n_actions = len(action_dict)
        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)
        self.action_dict = action_dict
        self.strategy = np.repeat(1/self.n_actions, self.n_actions)
        self.reach_pr = 0
        self.reach_pr_sum = 0
        self.player = player

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
    # print()
    # print('player 0 strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 0 and 'F' not in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    # print()
    # print('player 1 strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 1 and 'F' not in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    # print()
    # print('player 2 strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 2 and 'F' not in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    # print()
    # print('player 0 FLOP strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 0 and 'F' in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    # print()
    # print('player 1 FLOP strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 1 and 'F' in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    # print()
    # print('player 2 FLOP strategies:')
    # for _, v in filter(lambda x: int(x[0][0]) == 2 and 'F' in x[0], sorted_items):
    #     print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))

def save_results(i_map):
    lineas = []
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])

    lineas.append('\n')
    lineas.append('player 0 strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 0 and 'F' not in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum) + '\n')
    lineas.append('\n')
    lineas.append('player 1 strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 1 and 'F' not in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum) + '\n')
    lineas.append('\n')
    lineas.append('player 2 strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 2 and 'F' not in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum) + '\n')
    lineas.append('\n')
    lineas.append('player 0 FLOP strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 0 and 'F' in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum)  + '\n')
    lineas.append('\n')
    lineas.append('player 1 FLOP strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 1 and 'F' in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum)  + '\n')
    lineas.append('\n')
    lineas.append('player 2 FLOP strategies:' + '\n')
    for _, v in filter(lambda x: int(x[0][0]) == 2 and 'F' in x[0], sorted_items):
        lineas.append(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum)  + '\n')
    now = datetime.now()
    basePath = 'C:/temp2/leducFull/leducFULL3_2.txt'
    with open(basePath, 'w+') as archivo:
        archivo.seek(0)
        archivo.writelines(lineas)
        archivo.close()


if __name__ == "__main__":
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=1000)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))
