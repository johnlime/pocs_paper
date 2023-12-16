from env import *

env = EAGymEnvWrapper("CartPole-v1")
print(env.get_fitness(np.random.rand(env.genome_size), render = True))
