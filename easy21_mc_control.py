# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:25:11 2018

@author: Pan Zhao

The codes at https://github.com/analog-rl/Easy21 were referred to in creating this code
"""

#%% 
import os 
import sys

# get the directory and add its path to python search path for modules
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirpath)

import numpy as np
from easy21 import *
import random
import matplotlib.cm as cm

# dealer card value: 1-10, player sum: 1-21

# note the state S consists of the dealer's showing card and the player's sum 
# initialization 

class MC_Agent:
    def __init__(self,environment,N0):
        self.N0 = N0; 
        self.env = environment;
        # Intialize state value function to zero
        self.V= np.zeros((self.env.dealer_value_count,self.env.player_value_count))
        # Initialize state-action pair to zero
        self.Q = np.zeros((self.env.dealer_value_count,self.env.player_value_count,self.env.action_count))
        # N(s,a) is the number of times that action a has been selected from state s.
        self.N = np.zeros((self.env.dealer_value_count,self.env.player_value_count,self.env.action_count)) 
       
        # return at each state-action pair, using a list of list of list 
        self.G = [];
        for k in range(self.env.dealer_value_count*self.env.player_value_count):
            G1 =[];
            for j in range (self.env.action_count):
                G1.append([])
            self.G.append(G1)
        
        self.episodes = 0
        self.count_wins = 0

    # selecting an action according to the epsilon-greedy policy 
    def select_action(self,state):
        dealer_id = state.dealer -1;
        player_id = state.player - 1;
        
        epsilon = self.N0/(self.N0+sum(self.N[dealer_id, player_id,:]))
        
        if random.random()< epsilon:
            if random.random() < 0.5:
                action = Action.hit;
            else:
                action = Action.stick
        else:
            action = Action.to_action(np.argmax(self.Q[dealer_id, player_id,:]))
            
        return action    
                
    def train(self,episodes):    
        for episode in range(episodes):
            episode_pairs = []
            # random start 
            s = self.env.gen_start_state()
            
            # generate an episode with the epsilon-greedy policy
            while not s.is_terminal:
                a = self.select_action(s)       
                # update N(s,a)
                self.N[s.dealer-1,s.player-1,Action.as_int(a)] += 1
                
                # store action-value pairs
                episode_pairs.append((s,a))
                # execute action
                s, r= self.env.step(s,a)
            
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins 
                
            for s,a in episode_pairs:
                dealer_id = s.dealer-1
                player_id = s.player-1
                # update the state-action-return pair
                idx = dealer_id*10+player_id
                self.G[idx][Action.as_int(a)].append(r)
                
                # update Q(s,a) using a varying step size alpha = 1/N(st,at)
                alpha = 1.0/self.N[dealer_id,player_id,Action.as_int(a)] 
                error = np.mean(self.G[idx][Action.as_int(a)]) - self.Q[dealer_id,player_id,Action.as_int(a)]
                self.Q[dealer_id,player_id,Action.as_int(a)] += alpha*error
                
                # update V(s)
                # ideally update of V(s) should happen only ONCE after update of Q(s,a) for all s,a in the episode;
                # but for coding simplifity, it is updated right after every update of Q(s,a). This should not influence the final results. 
                self.V[dealer_id,player_id] = max(self.Q[dealer_id,player_id,:])
        self.episodes = self.episodes + episodes      
    def plot_frame(self, ax):
        def get_state_val(x,y):
            return self.V[x,y]
        X = np.arange(0,self.env.dealer_value_count,1)
        Y = np.arange(0,self.env.player_value_count,1)
        X,Y = np.meshgrid(X,Y)
        Z = get_state_val(X,Y)
        
        surf = ax.plot_surface(X,Y,Z,cmap=cm.bwr,antialiased=False)
        return surf
        

#%% Train and generate the value function 
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
#
#
#N0 = 100
#agent = MC_Agent(Environment(),N0)
#for i in range(1):
#    agent.train(50000)
#    
#fig = plt.figure()
#ax = fig.add_subplot(111,projection ='3d')
#agent.plot_frame(ax)
#plt.title('value function after %d episodes' % agent.episodes)
#ax.set_xlabel('Dealer showing')
#ax.set_ylabel('Player sum')
#ax.set_zlabel('V(s)')
#ax.set_xticks(range(1,agent.env.dealer_value_count+1))
#ax.set_yticks(range(1,agent.env.player_value_count+1))
#plt.show()
#plt.savefig('Value function.png')
#    

#%% animate the learning process 
    #import matplotlib.animation as animation
#def update(frame):
#    agent.train(10000)
#    
#    ax.clear()
#    surf = agent.plot_frame(ax)
#    plt.title('winning perc: %f frame:%s ' % (float(agent.count_wins)/agent.episodes,frame))
#    fig.canvas.draw()
#    return surf
#    
#
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
#N0 = 100
#agent = MC_Agent(Environment(),N0)
##fig=plt.figure('N0=%d' % N0)
#fig = plt.figure('N100')
#ax = fig.add_subplot(111,projection ='3d')
#ani = animation.FuncAnimation(fig,update,4,repeat=False)
#
#ani.save('MC_process.gif',writer = 'imagemagick',fps=3)
#plt.show()
# show the gif
#from IPython.display import Image
#Image(url="MC_process.gif")