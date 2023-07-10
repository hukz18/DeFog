import gym
import torch
import d4rl_atari
import numpy as np
from dotmap import DotMap
from buffer import AtariBuffer
import timeit

import numpy as np
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TorchDeque:
    def __init__(self, maxlen, device):
        self.maxlen = maxlen
        self.device = device
        self.deque = deque(maxlen=maxlen)

    def append(self, array:torch.Tensor):
        if len(self.deque) == self.maxlen:
            self.deque.popleft() # Remove the oldest array
        self.deque.append(array.to(self.device)) # Add the new array

    def to_numpy(self):
        # Converts the deque of 1D arrays into a 2D array
        return torch.stack(list(self.deque), dim=1).to(self.device)

class CircularBuffer:
    def __init__(self, maxlen, width):
        self.buffer = np.zeros((maxlen, *width))
        self.maxlen = maxlen
        self.width = width
        self.start = 0
        self.length = 0

    def append(self, array):
        if self.length < self.maxlen:
            self.buffer[self.length] = array
            self.length += 1
        else:
            self.buffer[self.start] = array
            self.start = (self.start + 1) % self.maxlen

    def to_numpy(self):
        if self.start == 0 or self.length < self.maxlen:
            return self.buffer[:self.length]
        return np.concatenate((self.buffer[self.start:], self.buffer[:self.start]))

def format_time(t):
    units = ['s', 'ms', 'µs', 'ns']
    thresholds = [1, 1e-3, 1e-6, 1e-9]

    for unit, threshold in zip(units, thresholds): # type: ignore
        if t >= threshold:
            return f"{t/threshold:.3f}{unit}"
    return f"{t:.3f}{units[-1]}"

def timeit_func(func, name, calls=10):
    timer = timeit.Timer(func)
    call_times = []
    for _ in range(calls):
        times, total_time = timer.autorange()
        call_times.append(total_time / times)
    print(f'{name}: {format_time(np.mean(call_times))} ± {format_time(np.std(call_times))}')

if __name__ == '__main__':
    # buffer = AtariBuffer('breakout', 'expert', 20, 4, 1, DotMap({'drop_fn': 'const', 'drop_p': 0.8, 'update_interval': 500, 'drop_aware': True}))
    # buffer.sample(16)
    # print(buffer)
    # Usage
    # buffer = CircularBuffer(20, (4, 84, 84))
    # buffer.append(np.random.rand(10, 4, 84, 84))
    # timeit_func(lambda: buffer.append(np.random.rand(10, 4, 84, 84)), 'append')
    # timeit_func(lambda: buffer.to_numpy(), 'to_numpy')
    # Usage
    context_len = 20
    batch_size = 64
    stack_frame = 4
    env = gym.make('breakout-expert-v0', stack=stack_frame)
    ds = env.get_dataset()
    selected_sp = np.random.choice(len(ds['observations']) - context_len, batch_size)
    torch.stack([torch.from_numpy(np.stack(ds['observations'][sp: sp+context_len])) for sp in selected_sp]).to(dtype=torch.float32, device=device)
    timeit_func(lambda: torch.stack([torch.from_numpy(np.stack(ds['observations'][sp: sp+context_len])) for sp in selected_sp]).to(dtype=torch.float32, device=device), 'fetch and to')
    timeit_func(lambda: torch.stack([torch.from_numpy(np.stack(ds['observations'][sp: sp+context_len])) for sp in selected_sp]).to(dtype=torch.float32, device=device, non_blocking=True), 'fetch and to non blocking')
    timeit_func(lambda: torch.stack([torch.from_numpy(np.stack(ds['observations'][sp: sp+context_len])).to(dtype=torch.float32, device=device) for sp in selected_sp]), 'to and fetch')
    timeit_func(lambda: torch.stack([torch.from_numpy(np.stack(ds['observations'][sp: sp+context_len])).to(dtype=torch.float32, device=device, non_blocking=True) for sp in selected_sp]), 'to and fetch non blocking')
    
    # buffer = TorchDeque(20, torch.device('cuda'))
    # buffer.append(torch.rand(10, 4, 84, 84))
    # buffer.append(torch.rand(10, 4, 84, 84))
    # buffer.append(torch.rand(10, 4, 84, 84))
    # buffer.append(torch.rand(10, 4, 84, 84))
    # print(buffer.to_numpy())
    # timeit_func(lambda: buffer.append(torch.rand(10, 4, 84, 84)), 'append')
    # timeit_func(lambda: buffer.to_numpy(), 'to_numpy')