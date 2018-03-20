# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:12:25 2018

@author: Boran Zhao

Reference:
https://github.com/analog-rl/Easy21
"""
#%% 
import random
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
class Action(Enum):
    stick = 0
    hit = 1    
    
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

def step (state, action):
    new_state = state;
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

#%% Test: whether draws follow a prespecified distribution
cards = [None]*10000;
for i in np.arange(len(cards)):
    cards[i] = Card()

f = plt.figure(1);

plt.subplot(2,1,1)
plt.title('Test: the value of the cards follow a uniform distribution \n between 1 and 10')
card_value = [abs(card.value) for card in cards]
plt.hist(card_value)
plt.subplot(2,1,2)
plt.title('Test: the probability of a red card (0) and a black card (1) \n are 1/3 and 2/3, respectively')
card_is_black = [card.is_black for card in cards]
plt.hist(card_is_black)
plt.subplots_adjust(hspace = 0.6 ) # adjust the horizontal space between subplots
#plt.show()
plt.savefig("test1.png");
#%% Test: If the player’s sum exceeds 21, or becomes less than 1, then he “goes bust” and loses the game (reward -1)
def test_player_bust():
    state = State(Card(True).value, Card(True).value)
    action = Action.hit
    while not state.is_terminal:
        state, reward = step(state,action)
    return state, reward

import matplotlib.pyplot as plt
plt.figure(2)
values = [];
for i in range(10000):
    state,reward = test_player_bust()
    if state.player > 21 or state.player < 1:
        values.append(1)
    else:
        values.append(-1)
        print("error! The player's score should be >21 or <1 at the terminal state!")

plt.hist(values)
plt.title("Test: player bust >21 or <1")
plt.savefig("test2.png")


#%% Test: player stick. 

def test_player_stick():
    state = State(Card(True).value, Card(True).value)
    action = Action.stick
    while not state.is_terminal:
        state, reward = step(state,action)
    return state, reward

import matplotlib.pyplot as plt

values = []
for i in range(10000):
    state,reward = test_player_stick()
    if state.dealer > 21 or state.dealer < 1:
        if reward == 1:
            values.append(1)
        else:
            values.append(-1)
            print('error! The player should have won!')
    elif state.player > state.dealer:
        if reward == 1:
            values.append(1)
        else:
            values.append(-2)
            print('error! The player should have won!')
    elif state.player == state.dealer:
        if reward == 0:
            values.append(1)
        else:
            values.append(-3)
            print('error! It should be a tie!')
            
plt.figure(3)
plt.hist(values)
plt.title("Test: player stick")
plt.savefig("test3.png")


#%% play 
def play():
    dealer = Card(True).value
    player = Card(True).value
    state = State(dealer,player)
    reward = 0
    while not state.is_terminal:
#        print('dealer score:%d, player score:%d' %  (state.dealer, state.player))
        if state.player > 17:
            action = Action.stick
        else:
            action = Action.hit
        state, reward = step(state,action)
#    print('dealer score:%d, player score:%d' %  (state.dealer, state.player))   
    print("reward = %d" % reward);
    return reward

