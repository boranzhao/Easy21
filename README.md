# Easy21



1 Implementation of Easy21: easy21-implement.py

2 Monte-Carlo Control in Easy21: easy21_mc_control.py

3 TD Learning in Easy21: easy21_sarsa_lambda.py

4 Linear Function Approximation in Easy21: easy21_sarsa_lambda_approx

5 Discussion

What are the pros and cons of bootstrapping in Easy21?<br>
Pros: no need to wait until the end of an episode, accelerate the learning process, decrease the variance <br>
Cons: may introduce bias 


Would you expect bootstrapping to help more in blackjack or Easy21 ? Why? <br>
Help more in Easy21, because it takes a longer time on average to finish an episode in easy21 due to the fact that a value of a card can be negative depending on its color.

What are the pros and cons of function approximation in Easy21?<br>
Pros: memory saving, learning speed acceleration<br>
Cons: can only solve the problem approximately since a function approximator cannot represent all the state-action values accurately 

How would you modify the function approximator suggested in this section to get better results in Easy21?
Have no idea right now. 
