import gym
from Agent import *
import numpy as np 

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(discount_factor=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_min=0.01, input_dims=[8], lr=0.01)
    
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0]
        while not done:
            action = agent.choose_action(observation)
            new_obs, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, new_obs, done)
            agent.learn()
            observation = new_obs
        
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(f'Episode -- {i}, score = {score}, average Score = {avg_score}, epsilon = {agent.epsilon}')
