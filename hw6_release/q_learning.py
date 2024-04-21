import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import gym

# TODO: implement this class
class QAgent:
    def __init__(self, state_space, action_space, init_val=0):
        """Initialize table for Q-value to init_val.
        The table should have shape [n_states, n_actions].
        state_space: env.observation_space
        action_space: env.action_space
        """
        raise NotImplementedError
    
    def act(self, state, epsilon, train=False, action_mask=None):
        if train:
            return self._act_train(state, epsilon, action_mask)
        else:
            return self._act_eval(state, epsilon, action_mask)
    
    def _act_train(self, state, epsilon, action_mask=None):
        """Implement epsilon-greedy strategy for action selection in training.
            i.e., with probability epsilon you should take a random action and 
            otherwise pick the best action according to the Q table.
        NOTE: you can ignore action_mask until part 6
        """
        raise NotImplementedError
    
    def _act_eval(self, state, epsilon, action_mask=None):
        """Implement action selection in evaluation (i.e., always pick best action).
        """
        raise NotImplementedError
    
    def update(self, state, action, reward, next_state, alpha, gamma, next_state_action_mask=None):
        """Implement Q-value table update here.
        NOTE: you may ignore the action mask until part 6
        """
        raise NotImplementedError

# -----------------------------------------------------------------------------
# NOTE: you do not need to modify the 3 functions below...
#       though you should do so for debugging purposes
# -----------------------------------------------------------------------------

def train_agent(env : gym.Env, agent : QAgent, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, use_action_mask=False):
    all_ep_rewards = []
    episode_reached_goal_rate = 0
    for i in tqdm.tqdm(range(epochs)):
        # each episode begins with a reset
        state, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # choose action
            if use_action_mask:
                action = agent.act(state, epsilon, train=True, action_mask=info["action_mask"])
            else:
                action = agent.act(state, epsilon, train=True)
            # act in the environment
            next_state, reward, term, trunc, info = env.step(action) 
            done = term or trunc
            # update agent (Q-learning)
            if use_action_mask:
                agent.update(state, action, reward, next_state, alpha, gamma, next_state_action_mask=info["action_mask"])
            else:
                agent.update(state, action, reward, next_state, alpha, gamma)

            state = next_state
            ep_reward += reward

        if reward > 0: 
            # use last reward instead of ep_reward because agent may have wandered for a long time
            episode_reached_goal_rate += 1
        all_ep_rewards.append(ep_reward)
        
    print("Training finished.\n")
    print("Episode success rate", episode_reached_goal_rate / epochs)
    
    return all_ep_rewards

def eval_agent(env : gym.Env, agent : QAgent, epsilon=0.1):
    state, info = env.reset()
    all_rewards = 0.0
    all_frames = [env.render()] # NOTE: assumes 'ansi' render_mode
    done = False
    
    while not done:
        action = agent.act(state, epsilon, train=False)
        
        state, reward, term, trunc, _ = env.step(action) 
        done = term or trunc
        
        all_frames.append(env.render())

        all_rewards += reward

    print(f"Obtained total reward of {np.sum(all_rewards)} after {len(all_frames)} steps")

    for frame in all_frames:
        print(frame)

def q_learning(alpha=0.1, gamma=0.9, epsilon=0.1, init_val=0, use_action_mask=False, save_path="train_rewards.png"):
    # create the environment, set the render_mode
    env = gym.make("Taxi-v3", render_mode="ansi")
    # initialize our agent
    agent = QAgent(env.observation_space, env.action_space, init_val=init_val)
    # train
    all_ep_rewards = train_agent(env, agent, alpha=alpha, gamma=gamma, epsilon=epsilon, use_action_mask=use_action_mask)
    # create moving average plot
    N = 10
    mov_avg_ep_rewards = np.convolve(np.array(all_ep_rewards), np.ones(N) / N, mode='valid')
    plt.plot(mov_avg_ep_rewards[:1000])
    plt.xlabel("Epochs")
    plt.ylabel("Episode Returns")
    plt.savefig(save_path)
    plt.clf()

    eval_agent(env, agent, epsilon=epsilon)


if __name__ == "__main__":
    # We've filled in the experiment you should run for part b
    experiments = {
        "5b": {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.1, 
            "init_val": 0.0,
            "use_action_mask": False
        }
    }
    for exp_name in experiments:
        q_learning(**experiments[exp_name], save_path=f"train_rewards_{exp_name}.png")