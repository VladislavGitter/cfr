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
        self.deck = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
        self.action_dict = {}

        self.traverser = 0

    def train(self, n_iterations=50000):
        expected_game_value = 0
        under5counter = 0
        for _ in range(n_iterations):
            # self.traverser = 0 if _ % 2 == 0 else 1
            log('[Kuhn.train]-ITERATION START')
            shuffle(self.deck)
            log('[Kuhn.train]-deck: ' + str(self.deck))
            expected_game_value += self.cfr('', 1, 1, '')
            if abs(expected_game_value/_)<=0.05:
                under5counter += 1
            else:
                under5counter = 0
            if _ % 1 == 0: print(str(_) + '[Kuhn.train]-expected_game_value: ' + str(expected_game_value/_) + ' <0.05: ' + str(under5counter))
            for _, v in self.nodeMap.items():
                # log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + '): ' + str(v.strategy))
                v.update_strategy()
            log('[Kuhn.train]-ITERATION END')
            log('=======================================')
        for _, v in self.nodeMap.items():
            log('[Kuhn.train]-UPDATE STRATEGY' + '(' + str(v.key) + ')' + str(v.action_dict) + ': ' + str(v.strategy))
        expected_game_value /= n_iterations
        # display_results(expected_game_value, self.nodeMap)
        save_results(self.nodeMap)

    def cfr(self, history, pr_1, pr_2, logStr):
        log(logStr + '[Kuhn.cfr]-START!!, history: ' + history + ' pr_1:' + str(pr_1) + ' pr_2:' + str(pr_2))
        
        self.action_dict = self.get_action_dict(history)
        n_actions = len(self.action_dict)
        active_history = history[history.index('F')+1:] if 'F' in history else history
        active_history = active_history[active_history.index('T')+1:] if 'T' in active_history else active_history
        active_history = active_history[active_history.index('R')+1:] if 'R' in active_history else active_history
        n = len(active_history)
        is_player_1 = n % 2 == 0
        # player = n % 2
        plStr = 'P1' if is_player_1 else 'P2'
        log(logStr + '[Kuhn.cfr]-player1: ' + str(is_player_1))
        player_card = self.deck[0] if is_player_1 else self.deck[1]
        log(logStr + '[Kuhn.cfr]-player_card: ' + str(player_card))
        if self.is_terminal(history):
            card_player = self.deck[0] if is_player_1 else self.deck[1]
            card_opponent = self.deck[1] if is_player_1 else self.deck[0]
            reward = self.get_reward( history, card_player, card_opponent, self.deck[2:4])
            log(logStr + '[Kuhn.cfr]-TERMINAL, reward: ' + str(reward))
            return reward
        
        
        if self.start_flop(history):
            is_player_1 = True
            player_card = self.deck[0]
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            plStr = 'P1'
            pot = self.get_pot(history)
            flopStr = str(pot) + 'F'
            history = flopStr
        elif self.start_turn(history):
            is_player_1 = True
            player_card = self.deck[0]
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            plStr = 'P1'
            pot = self.get_pot(history)
            turnStr = str(pot) + 'T'
            history = turnStr
        elif self.start_river(history):
            is_player_1 = True
            player_card = self.deck[0]
            self.action_dict = {0: 'p', 1: 'b'}
            n_actions = len(self.action_dict)
            plStr = 'P1'
            pot = self.get_pot(history)
            riverStr = str(pot) + 'R'
            history = riverStr
            

        board_cards = self.get_bd_cards_state(history)
        
        node = self.get_node(player_card,board_cards, history)
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
        log(logStr + '[Kuhn.cfr]-calculated c. utils: ' + str(action_utils))
        # Utility of information set.
        util = sum(action_utils * strategy)
        log(logStr + '[Kuhn.cfr]-util(action_utils*strategy): ' + str(util))
        regrets = action_utils - util
        log(logStr + '[Kuhn.cfr]-regrets(action_utils-util): ' + str(regrets))
        log(logStr + '[Kuhn.cfr]-updating regret_sum, reach_pr...')
        # if self.traverser == player:
        if is_player_1:
            node.reach_pr += pr_1
            node.regret_sum += pr_2 * regrets
        else:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regrets
        log(logStr + '[Kuhn.cfr]-updated regret_sum, reach_pr: ' + str(node.regret_sum) + ', ' +  str(node.reach_pr))
        log(logStr + '[Kuhn.cfr]-END!!, history: ' + history + ' return:' + str(util))
        return util
    
    def get_bd_cards_state(self, history):
        board_cards = []
        if 'R' in history:
            board_cards.append(self.deck[2])
            board_cards.append(self.deck[3])
            board_cards.append(self.deck[4])
            return board_cards
        elif 'T' in history:
            board_cards.append(self.deck[2])
            board_cards.append(self.deck[3])
            return board_cards
        elif 'F' in history:
            board_cards.append(self.deck[2])
            return board_cards
        return []

    @staticmethod
    def get_action_dict(history):
        history = history[history.index('F')+1:] if 'F' in history else history
        history = history[history.index('T')+1:] if 'T' in history else history
        history = history[history.index('R')+1:] if 'R' in history else history
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
        in_river = 'R' in history
        if history[-2:] == 'bp':
            return True
        elif in_river and (history[-2:] == 'pp' or history[-2:] == "bc"):
            return True
    @staticmethod
    def start_river(history):
         return ('R' not in history and (history[-2:] == 'pp' or history[-2:] == "bc"))
    @staticmethod
    def start_turn(history):
         return ('T' not in history and 'R' not in history and (history[-2:] == 'pp' or history[-2:] == "bc"))
    @staticmethod
    def start_flop(history):
         return ('F' not in history and 'T' not in history and 'R' not in history and (history[-2:] == 'pp' or history[-2:] == "bc"))
    def get_reward(self, history, player_card, opponent_card, board_cards):
        pot = self.get_pot(history)
        terminal_pass = history[-1] == 'p'
        bet_call = history[-2:] == "bc"
        in_river = 'R' in history
        winner = self.get_winner_showdown(player_card, opponent_card,board_cards)
        if terminal_pass:
            if history[-2:] == 'pp' and in_river:
                if winner == 1: return pot/2
                elif winner == -1: return -pot/2
                else: return pot/4
            else: #bet_pass
                # subtract 2 from pot as the other player doesnt call, so the last bet/raise doesnt count for reward
                return (pot-2)/2
        elif bet_call and in_river:
            if winner == 1: return pot/2
            elif winner == -1: return -pot/2
            else: return pot/4
    
    def get_winner_showdown(self, player_card, opponent_card, board_cards):
        # returns 1 if player wins, 0 draw, -1 if opponent wins
        #leduc full version(preflop-flop-turn-river), checks sets, 

        pl_hand_score = self.get_hand_score(player_card, board_cards)
        opp_hand_score = self.get_hand_score(opponent_card, board_cards)
        winner = self.compare_scores(pl_hand_score, opp_hand_score)
        return winner
    @staticmethod
    def compare_scores(pl_hand_score, opp_hand_score):
        if pl_hand_score[0] > opp_hand_score[0]:
            return 1
        elif pl_hand_score[0] < opp_hand_score[0]:
            return -1
        elif pl_hand_score[0] == opp_hand_score[0] and pl_hand_score[0] != -1:
            if pl_hand_score[3] > opp_hand_score[3]:
                return 1
            if pl_hand_score[3] < opp_hand_score[3]:
                return -1
            else:
                return 0
        if pl_hand_score[1] > opp_hand_score[1]:
            return 1
        elif pl_hand_score[1] < opp_hand_score[1]:
            return -1
        elif pl_hand_score[1] == opp_hand_score[1] and pl_hand_score[1] != -1:
            return 0
        if pl_hand_score[2] > opp_hand_score[2]:
            return 1
        elif pl_hand_score[2] < opp_hand_score[2]:
            return -1
        elif pl_hand_score[2] == opp_hand_score[2] and pl_hand_score[2] != -1:
            if pl_hand_score[3] > opp_hand_score[3]:
                return 1
            if pl_hand_score[3] < opp_hand_score[3]:
                return -1
            else:
                return 0
        if pl_hand_score[3] > opp_hand_score[3]:
            return 1
        if pl_hand_score[3] < opp_hand_score[3]:
            return -1
        else:
            return 0

     
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

    @staticmethod 
    def get_pot(history):
        pot = 2 # 2 blinds, 1bb each
        to_call = 0
        # IR
        if 'F' in history:
            pot = int(history[:history.index('F')])
        elif 'T' in history:
            pot = int(history[:history.index('T')])
        elif 'R' in history:
            pot = int(history[:history.index('R')])
        for action in history:
            if action == 'b':
               pot += 2 + to_call
               to_call = 2
            elif action == 'c':
                pot += to_call
                to_call = 0
        # print('h:' + history + ' pot: ' + str(pot))
        return pot

    def get_node(self, pl_card, bd_cards, history):
        cardsStr = '(' +  str(pl_card) + ','
        for c in bd_cards:
            cardsStr = cardsStr + str(c) +','
        cardsStr = cardsStr.removesuffix(',') + ')'
        key = cardsStr + history
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
    
    print()
    print('player 1 FLOP strategies:')
    for _, v in filter(lambda x: 'F' in x[0] and 'T' not in x[0] and len(x[0][x[0].index('F') + 2:]) % 2 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 FLOP strategies:')
    for _, v in filter(lambda x: 'F' in x[0] and 'T' not in x[0] and len(x[0][x[0].index('F') + 2:]) % 2 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))

    print()
    print('player 1 TURN strategies:')
    for _, v in filter(lambda x: 'T' in x[0] and 'R' not in x[0] and len(x[0][x[0].index('T') + 2:]) % 2 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 TURN strategies:')
    for _, v in filter(lambda x: 'T' in x[0] and 'R' not in x[0] and len(x[0][x[0].index('T') + 2:]) % 2 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    
    print()
    print('player 1 RIVER strategies:')
    for _, v in filter(lambda x: 'R' in x[0] and len(x[0][x[0].index('R') + 2:]) % 2 == 1, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))
    print()
    print('player 2 RIVER strategies:')
    for _, v in filter(lambda x: 'R' in x[0] and len(x[0][x[0].index('R') + 2:]) % 2 == 0, sorted_items):
        print(str(v) +' '+ str(v.regret_sum) + ' %:' + str(v.reach_pr_sum))

def save_results(i_map):
    lineas = []
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    lineas.append('\n')
    lineas.append('p1 PFLOP:' + '\n')
    for _, v in filter(lambda x: len(x[0]) % 2 == 0 and 'F' not in x[0] and 'T' not in x[0] and 'R' not in x[0], sorted_items):
        lineas.append(str(v) + '\n')
    lineas.append('\n')
    lineas.append('p2 PFLOP:' + '\n')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1 and 'F' not in x[0] and 'T' not in x[0] and 'R' not in x[0], sorted_items):
        lineas.append(str(v) + '\n')

    lineas.append('\n')
    lineas.append('p1 FLOP:' + '\n')
    for _, v in filter(lambda x: 'F' in x[0] and len(x[0][x[0].index('F'):]) % 2 == 1, sorted_items):
        lineas.append(str(v) + '\n')
    lineas.append('\n')
    lineas.append('p2 FLOP:' + '\n')
    for _, v in filter(lambda x: 'F' in x[0] and len(x[0][x[0].index('F'):]) % 2 == 0, sorted_items):
        lineas.append(str(v) + '\n')

    lineas.append('\n')
    lineas.append('p1 TURN:' + '\n')
    for _, v in filter(lambda x: 'T' in x[0] and len(x[0][x[0].index('T'):]) % 2 == 1, sorted_items):
       lineas.append(str(v) + '\n')
    lineas.append('\n')
    lineas.append('p2 TURN:' + '\n')
    for _, v in filter(lambda x: 'T' in x[0] and len(x[0][x[0].index('T'):]) % 2 == 0, sorted_items):
        lineas.append(str(v) + '\n')
    
    lineas.append('\n')
    lineas.append('p1 RIVER:' + '\n')
    for _, v in filter(lambda x: 'R' in x[0] and len(x[0][x[0].index('R'):]) % 2 == 1, sorted_items):
        lineas.append(str(v) + '\n')
    lineas.append('\n')
    lineas.append('p2 RIVER:' + '\n')
    for _, v in filter(lambda x: 'R' in x[0] and len(x[0][x[0].index('R'):]) % 2 == 0, sorted_items):
        lineas.append(str(v) + '\n')

    with open('C:/temp2/leducFull/leducFullIR4.txt', 'w+') as archivo:
        archivo.seek(0)
        archivo.writelines(lineas)
        archivo.close()


if __name__ == "__main__":
    time1 = time.time()
    trainer = Kunh()
    trainer.train(n_iterations=30)
    print(abs(time1 - time.time()))
    print(sys.getsizeof(trainer))

