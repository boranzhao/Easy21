# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:21:25 2018

@author: Pan Zhao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:12:25 2018

@author: Boran Zhao

The codes at https://github.com/analog-rl/Easy21 were referred to in creating this code
"""
#%% 
import random
import matplotlib.pyplot as plt
import copy

from enum import Enum
class Action(Enum):
    hit = 0    
    stick = 1
    
    
    @staticmethod
    def to_action(n):
        return Action.hit if n==0 else Action.stick
        
    @staticmethod
    def as_int(a):
        return 0 if a==Action.hit else 1 
    
    
    
class Card:
    def __init__(self,force_black = False):
        self.value = random.randint(1,10)        
        if force_black or random.randint(1,3) != 3:
            self.is_black = True;            
        else:
            self.is_black = False;
            self.value = -self.value;


class State:
    def __init__ (self, dealer, player, is_terminal = False):        
        self.dealer = dealer    # the summation of the dealer
        self.player = player    # the summation of the player 
        self.is_terminal = is_terminal # whether the state is terminal


class Environment:
    def __init__(self):
        self.dealer_value_count = 10; # [1:10], note that black card is enforced at the start
        self.player_value_count = 21; # [1:21]
        self.action_count = 2; # hit and stick
        
    def gen_start_state(self):
        s = State(Card(True).value,Card(True).value)
        return s
        
    def step(self, state, action):
#        new_state = state does not work because modifying new_state will influence state
        new_state = copy.copy(state)
        reward = 0;
        if action == Action.hit:
            new_state.player  += Card().value;
            if new_state.player > 21 or new_state.player <1:
                new_state.is_terminal = True;
                reward = -1
                return new_state, reward
        elif action == Action.stick:
            while not new_state.is_terminal:
                new_state.dealer += Card().value;
                if new_state.dealer > 21 or  new_state.dealer < 1:
                    new_state.is_terminal = True;
                    reward = 1
                elif new_state.dealer> 17:
                    new_state.is_terminal = True;
                    if new_state.player > new_state.dealer:
                        reward = 1
                    elif new_state.player < new_state.dealer:
                        reward = -1
        return new_state, reward