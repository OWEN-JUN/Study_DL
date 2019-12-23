import gym
from gym.envs.registration import register
import sys,tty,termios


class _Geetch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return ch


inkey = _Geetch()

left = 0
down = 1
right = 2
up = 3

arrow_keys = {'\x1b[A': up,
              '\x1b[B': down,
              '\x1b[C': right,
              '\x1b[D': left

              }

register(id="FrozenLake-v3", entry_point='gym.envs.toy_text:FrozenLakeEnv',kwargs={'map_name' : '4x4',
                                                                                   'is_slippery':False})

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("stste: ", state, "action: ", action, "reward: ",reward, "info: ",info)

    if done:
        print("finished with reward", reward)
        break