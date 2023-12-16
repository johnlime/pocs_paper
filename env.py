import gym
import numpy as np

class EAGymEnvWrapper:
    def __init__ (self, env_name, hidden_layer_size = 16):
        self.env_name = env_name
        self.h_size = hidden_layer_size
        env = gym.make(self.env_name)

        self.observation_dim, self.action_dim = 0, 0
        env_space_dim = [0, 0]
        self.output_activation = np.tanh
        for i, env_space in enumerate(
            [env.observation_space, env.action_space]):
            if env_space.__class__.__name__ == "Discrete":
                env_space_dim[i] = env_space.n
                if i == 1: # softmax function
                    self.output_activation = \
                        lambda x: np.argmax(
                            np.exp(x) / np.sum(np.exp(x)))
            elif env_space.__class__.__name__ == "Box":
                env_space_dim[i] = len(env_space.bounded_below)
            else:
                raise Exception("Space not implemented")

            self.observation_dim, self.action_dim = \
                env_space_dim[0], env_space_dim[1]

        env.close()
        self.genome_size = self.observation_dim * self.h_size + \
            self.h_size * self.h_size + \
            self.h_size * self.action_dim # genome size

    def get_fitness(self, genome):
        return self.run_gym(genome, render = False)[0]

    def run_gym(self, genome,
                render = False):
        assert len(genome) == self.genome_size

        env_render = lambda: 0
        if not render:
            env = gym.make(self.env_name) #, render_mode="human")
        else:
            env = gym.make(self.env_name) #, render_mode="human")
            env_render = env.render

        """
        Neural Net
        (observation_dim, 16, 16, action_dim)
        """
        index_tag = self.observation_dim * self.h_size
        layer_1 = np.reshape(genome[:index_tag],
            (self.h_size, self.observation_dim))

        layer_2 = np.reshape(
            genome[index_tag: index_tag + self.h_size * self.h_size],
            (self.h_size, self.h_size)
        )

        index_tag += self.h_size * self.h_size
        layer_3 = np.reshape(
            genome[index_tag:],
            (self.action_dim, self.h_size)
        )

        try:
            observation, info = env.reset()
        except:
            observation = env.reset()
        return_value = 0
        return_obs = np.empty((1000, self.observation_dim))
        truncated = False
        for t in range(1000):
            action = np.tanh(np.matmul(layer_1, observation))
            action = np.tanh(np.matmul(layer_2, action))
            action = self.output_activation(np.matmul(layer_3, action))
            try:
                observation, reward, terminated, truncated, info = \
                    env.step(action)
            except:
                observation, reward, terminated, info = env.step(action)
            return_value += reward
            return_obs[t] = np.array(observation)
            env_render()
            if terminated or truncated:
                try:
                    observation, info = env.reset()
                except:
                    observation = env.reset()

        env.close()

        return return_value, return_obs
