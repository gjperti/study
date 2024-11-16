import gymnasium as gym

env = gym.make('LunarLander-v3',render_mode="rgb_array")

env.reset()

for i in range(10):
    env.render()
    env.step(env.action_space.sample())
    
env.close()