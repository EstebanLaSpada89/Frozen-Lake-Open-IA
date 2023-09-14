import gym
import numpy as np
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)

    q = np.zeros((env.observation_space.n, env.action_space.n)) if is_training else pickle.load(open('frozen_lake4x4.pkl', 'rb'))
    learning_rate, discount_factor, epsilon, epsilon_decay_rate = 0.9, 0.9, 1, 0.0001
    rewards_per_episode = []

    for episode in range(episodes):
        state, terminated, truncated, total_reward = env.reset()[0], False, False, 0

        while not terminated and not truncated:
            action = env.action_space.sample() if (is_training and np.random.rand() < epsilon) else np.argmax(q[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] += learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])

            state, total_reward = new_state, total_reward + reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        rewards_per_episode.append(episode) if total_reward == 1 else None

    env.close()

    if is_training:
        pickle.dump(q, open("frozen_lake4x4.pkl", "wb"))

    print("Episodios con recompensa alcanzada:", rewards_per_episode)

if __name__ == '__main__':
    run(10, is_training=False, render=True)
    #run(15000, is_training=True, render=False)
    
    