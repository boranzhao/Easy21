# -*- coding: utf-8 -*-
"""
Implementation of Sarsa(\lambda) for Easy21 

@author: Pan Zhao

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

# note the state S consists of the dealer's showing card and the player's sum 
# initialization 

class Sarsa_Agent:
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
        # N(s,a) is the number of times that action a has been selected from state s.
        self.N = np.zeros((self.env.dealer_value_count,self.env.player_value_count,self.env.action_count)) 
        # eligibility trace for every state-action pair
        self.E = np.zeros((self.env.dealer_value_count,self.env.player_value_count,self.env.action_count))
        
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
                
    def train(self,num_episodes):    
        for episode in range(num_episodes):
            # random start 
            s = self.env.gen_start_state()          
            
            # generate an episode with the epsilon-greedy policy and update Q(s,a) and E(s,a) in each step
            a = self.select_action(s)       
            while not s.is_terminal:
                # update N(s,a)
                self.N[s.dealer-1,s.player-1,Action.as_int(a)] += 1
                # execute action a and observe s_new, r
                s_new, r= self.env.step(s,a)
                dealer_id = s.dealer-1
                player_id = s.player-1
                if s_new.is_terminal:
                    Q_new = 0
                else: 
                    # select a new action a_new using policy dervied from Q
                    a_new = self.select_action(s_new)
                    dealer_id_new = s_new.dealer-1
                    player_id_new = s_new.player-1
                    Q_new = self.Q[dealer_id_new,player_id_new,Action.as_int(a_new)]
                # using a varying step size alpha = 1/N(st,at)
                alpha = 1.0/self.N[dealer_id,player_id,Action.as_int(a)] 
                # calculate TD error
                td_error = r + self.gamma*Q_new - self.Q[dealer_id,player_id,Action.as_int(a)]
                # update the eligibility trace 
                self.E[dealer_id,player_id, Action.as_int(a)] += 1
                #update the Q and E for all state-action pairs 
                self.Q += alpha*td_error*self.E
                self.E *= self.gamma*self.lam
#                for q,e in np.nditer([self.Q,self.E],op_flags =['readwrite']):
#                    q[...] = q + alpha*td_error*e
#                    e[...] = self.gamma*self.lam*e                
                # update s and 
                s = s_new
                if not s_new.is_terminal:
                    a = a_new                            
            self.count_wins = self.count_wins+1 if r==1 else self.count_wins                 
        # report the mean-squared error mean((Q(s,a)-Qmc(s,a))^2 and the winning percentage
        self.episodes = self.episodes + num_episodes       
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

N0 = 100
mc_agent = MC_Agent(Environment(),N0)
mc_agent.train(int(1e6))
print('After %s episodes, winning percentage:%f' % (mc_agent.episodes, mc_agent.count_wins/mc_agent.episodes))

fig = plt.figure(1)
ax = fig.add_subplot(111,projection ='3d')
mc_agent.plot_frame(ax)
plt.title('value function after %d episodes' % mc_agent.episodes)
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player sum')
ax.set_zlabel('V(s)')
ax.set_xticks(range(1,mc_agent.env.dealer_value_count+1))
ax.set_yticks(range(1,mc_agent.env.player_value_count+1))
plt.savefig('Value function from MC.png')
plt.show()
Qmc = mc_agent.Q

#%% Train with Sarsa(lambda) using different lambda while printing the MSE of Q 
Lambda = np.linspace(0,1,10)
fig = plt.figure('MSE under different lambda values')
mse = []
Color = ['b','g','r','c','m','y','k']
LineStyle =['-','--',':','-.']
for i in range(len(Lambda)):
    mse.append([])

# Learn and plot the result
for lam_id,lam in enumerate(Lambda):
#    print('lambda = %s'% lam)
    agent = Sarsa_Agent(Environment(),N0,lam)
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
plt.savefig('MSE against lambda')
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

plt.savefig('Learning process for lambda 0 and 1')
plt.show()

##%% animate the learning process (does not work at this moment)
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
## show the gif
##from IPython.display import Image
##Image(url="MC_process.gif")