# -*- coding: utf-8 -*-
"""
Implementation of Sarsa(\lambda) with binary features and linear function approximation for Easy21 
following the algorithm presented in Section~12.7 of [Sutton & Barto, 2017]
@author: Boran (Pan) Zhao

"""

#%% 
import os 
import sys

# get the directory and add its path to python search path for modules
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirpath)

import numpy as np
from easy21 import *
from easy21_mc_control import *
import random

# dealer card value: 1-10, player sum: 1-21

# note the state S consists of the dealer's showing card and the player's summation 
# initialization 

class Sarsa_Approx_Agent:
    def __init__(self,environment,N0,lam):
        self.N0 = N0; 
        self.lam = lam
        self.gamma = 1
        self.mse = float('inf')
        self.env = environment;
        # Intialize state value function to zero
        self.V= np.zeros((self.env.dealer_value_count,self.env.player_value_count))
        # Initialize state-action pair to zero
        self.Q = np.zeros((self.env.dealer_value_count,self.env.player_value_count,self.env.action_count))
        # weight vecotor 
        self.w = np.zeros((3,6,2))
         # eligibility trace for the weight vector
        self.E = np.zeros((3,6,2))
        self.features ={'dealer':[[1,4], [4,7],[7,10]],  'player':[[1, 6],[4, 9],[7,12],[10, 15],[13, 18],[16, 21]], 'action':[Action.hit,Action.stick]}     
        
        
        self.episodes = 0
        self.count_wins = 0
    # determine phi(s,a)
    def feature_eval(self,s,a):
        Phi = np.zeros((3,6,2))
        for dealer_interval_id, dealer_interval in enumerate(self.features['dealer']):
            if dealer_interval[0] <= s.dealer <= dealer_interval[1]:
                for player_interval_id, player_interval in enumerate(self.features['player']):
                    if player_interval[0] <= s.player <= player_interval[1]:
                        Phi[dealer_interval_id,player_interval_id,Action.as_int(a)]=1
        return Phi
                        
                        
    # selecting an action according to the epsilon-greedy policy 
    def select_action(self,state):
        dealer_id = state.dealer -1
        player_id = state.player - 1
        # epsilon for exploration
#        epsilon = self.N0/(self.N0+sum(self.N[dealer_id, player_id,:]))
        # use a constant epsilon for exploration
        epsilon = 0.05        
        if random.random()< epsilon:
            if random.random() < 0.5:
                action = Action.hit;
            else:
                action = Action.stick
        else:
            action = Action.to_action(np.argmax(self.Q[dealer_id, player_id,:]))            
        return action    
                
    def train(self,num_episodes):
        for episode in range(num_episodes):
            # random start 
            s = self.env.gen_start_state()          
            # reset the eligibility traces.... Is this really necessary?
            self.E = np.zeros((3,6,2))
            
            # generate an episode with the epsilon-greedy policy and update Q(s,a) and E(s,a) in each step
            a = self.select_action(s)       
            while not s.is_terminal:
                # update N(s,a)
#                self.N[s.dealer-1,s.player-1,Action.as_int(a)] += 1
                 # execute action a and observe s_new, r
                s_new, r= self.env.step(s,a)
                td_error = r                
                # identify the active features and update their corresponding eligibility trace elements
                Phi =self.feature_eval(s,a)
                td_error -= np.sum(self.w*Phi)
                 # accumulating traces
                self.E += Phi
                # replacing traces 
#                self.E = Phi                
#                for dealer_interval_id, dealer_interval in enumerate(self.features['dealer']):
#                    if dealer_interval[0] <= s.dealer <= dealer_interval[1]:
#                        for player_interval_id, player_interval in enumerate(self.features['player']):
#                            if player_interval[0] <= s.player <= player_interval[1]:
#                                td_error -= self.w[dealer_interval_id,dealer_interval_id,Action.as_int(a)]
#                                # accumulating traces
#                                self.E[dealer_interval_id,dealer_interval_id,Action.as_int(a)] += 1
#                                # replacing traces 
#                                self.E[dealer_interval_id,dealer_interval_id,Action.as_int(a)] = 1

                if not s_new.is_terminal:
                    # select a new action a_new using policy dervied from Q
                    a_new = self.select_action(s_new)                    
                    Phi = self.feature_eval(s_new,a_new)                    
                    td_error += np.sum(self.gamma*(self.w*Phi))                    
                                       
                # using a constant step size for exploration
                alpha = 0.05                           
                #update the weight vector and eligibility trace 
                self.w += alpha*td_error*self.E
                self.E *= self.gamma*self.lam 
                # update s and 
                s = s_new
                if not s_new.is_terminal:
                    a = a_new                            
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins                 
        # report the mean-squared error mean((Q(s,a)-Qmc(s,a))^2 and the winning percentage
        self.episodes = self.episodes + num_episodes       
       
        # update the Q function
        self.update_Q()
        
    
    def update_Q(self):
        for dealer_id in range(self.Q.shape[0]):
            for player_id in range(self.Q.shape[1]):
                for action_id in range(self.Q.shape[2]):
                    Phi = self.feature_eval(State(dealer_id+1,player_id+1), Action.to_action(action_id))
                    self.Q[dealer_id,player_id,action_id] = np.sum(Phi*self.w)
        
    def update_V(self):
        for dealer_id in range(self.V.shape[0]):
            for player_id in range(self.V.shape[1]):
                self.V[dealer_id,player_id] = max(self.Q[dealer_id,player_id,:])
                
                
    def plot_frame(self, ax):
        def get_state_val(x,y):
            return self.V[x,y]
        X = np.arange(0,self.env.dealer_value_count,1)
        Y = np.arange(0,self.env.player_value_count,1)
        X,Y = np.meshgrid(X,Y)
        Z = get_state_val(X,Y)
        
        surf = ax.plot_surface(X,Y,Z,cmap=cm.bwr,antialiased=False)
        return surf    


#%% Train and generate the Q function with Monte Carlo control
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
#
#N0 = 100
#mc_agent = MC_Agent(Environment(),N0)
#mc_agent.train(int(1e6))
#print('After %s episodes, winning percentage:%f' % (mc_agent.episodes, mc_agent.count_wins/mc_agent.episodes))
#
#fig = plt.figure(1)
#ax = fig.add_subplot(111,projection ='3d')
#mc_agent.plot_frame(ax)
#plt.title('value function after %d episodes' % mc_agent.episodes)
#ax.set_xlabel('Dealer showing')
#ax.set_ylabel('Player sum')
#ax.set_zlabel('V(s)')
#ax.set_xticks(range(1,mc_agent.env.dealer_value_count+1))
#ax.set_yticks(range(1,mc_agent.env.player_value_count+1))
##plt.savefig('Value function.png')
#plt.show()
#Qmc = mc_agent.Q

#%% Train with Sarsa(lambda) with linear function approximation using different lambda while printing the MSE of Q 
import matplotlib.pyplot as plt
Lambda = np.linspace(0,1,11)
fig = plt.figure('MSE under different lambda values')
mse = []
Color = ['b','g','r','c','m','y','k']
LineStyle =['-','--',':','-.']
for i in range(len(Lambda)):
    mse.append([])

# Learn and plot the result
for lam_id,lam in enumerate(Lambda):
#    print('lambda = %s'% lam)
    agent = Sarsa_Approx_Agent(Environment(),N0,lam)
    for i in range(1000):
        agent.train(1)    
        agent.mse = np.mean((agent.Q-Qmc)**2)
        mse[lam_id].append(agent.mse)      
    print('lambda = %s, MSE: %f, winning percentage:%f' % (agent.lam, agent.mse, agent.count_wins/agent.episodes))
    
X = list(range(1,len(mse[0])+1))    
fig = plt.figure('MSE against lambda')
plt.plot(Lambda, [x[-1] for x in mse])
plt.xlabel('lambda')
plt.ylabel('mean-squared error')
plt.savefig('MSE against lambda under linear approximation')
plt.show()

fig = plt.figure('Learning process')
plt.subplot(211)
plt.plot(X,mse[0],color = Color[0], linestyle=LineStyle[0%4])
plt.xlabel('episode')
plt.ylabel('MSE')
plt.title('lambda = 0')

plt.subplot(212)
plt.plot(X,mse[-1],color = Color[0], linestyle=LineStyle[0%4])
plt.xlabel('episode')
plt.ylabel('MSE')
plt.title('lambda = 1')
plt.savefig('Learning process for lambda 0 and 1 under linear approximation')
plt.show()
