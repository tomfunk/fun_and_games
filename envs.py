import gym
from gym import spaces
import numpy as np
import string
from termcolor import cprint

class SudokuEnv1(gym.Env):
    """reference https://github.com/MorvanZhou/sudoku/blob/master/sudoku.py"""

    def __init__(self, mask_rate=0.5, max_attempts=250, render=False, flatten=True):
        self.max_attempts = max_attempts
        self.render = render
        self.flatten = flatten
        self.mask_rate = mask_rate

        if self.flatten:
            self.observation_space = spaces.MultiDiscrete([10] * 9 * 9 + [2] * 9 * 9)
        else:
            self.observation_space = spaces.MultiDiscrete([[[[10] * 9] * 9], [[[2] * 9] * 9]])

        # x, y, value - coordinates on sudoku puzzle, value to place on space
        # NOTE action is value - 1 e.g. action (0, 0, 0) places a 1 in (0, 0)
        self.action_space = spaces.MultiDiscrete([9, 9, 9])

    def count_dupes(self, observation=None):
        if observation is None:
            observation = self.observation

        duplicates = 0
        for i in range(9): 
            # rows
            values, counts = np.unique(observation[i, :], return_counts=True)
            duplicates += sum(counts[(values > 0) & (counts > 1)] - 1)
            # columns
            values, counts = np.unique(observation[:, i], return_counts=True)
            duplicates += sum(counts[(values > 0) & (counts > 1)] - 1)
            
        for i in range(3):
            for j in range(3):
                values, counts = np.unique(observation[i * 3: i * 3 + 3, j * 3: j * 3 + 3], return_counts=True)
                duplicates += sum(counts[(values > 0) & (counts > 1)] - 1)
        return duplicates
    
    @property
    def full_observation(self):
        return np.stack((self.observation, self.start > 0))

    def _get_reward(self, changed):
        if not changed:
            return -10
        dup_count = self.count_dupes()
        if dup_count > 0:
            reward = -dup_count
        elif (self.observation == 0).sum() == 0:
            reward = 1000
        else:
            reward = int((self.observation[self.start == 0] != 0).sum())
        return reward

    def update_state(self, action):
        self.attempt += 1
        (x, y, value) = action
        if self.start[x, y] == 0 and self.observation[x, y] != value + 1:
            self.observation[x, y] = value + 1
            return True
        return False

    def _get_obs(self, action=None):
        if self.flatten:
            return self.full_observation.flatten()
        return self.full_observation
    
    def _get_info(self):
        return {
            'attempt': self.attempt,
        }
    
    def generate_puzzle(self):
        while True:
            n = 9
            solution = np.zeros((n, n), int)
            rg = np.arange(1, n + 1)
            solution[0, :] = np.random.choice(rg, n, replace=False)
            try:
                for r in range(1, n):
                    for c in range(n):
                        col_rest = np.setdiff1d(rg, solution[:r, c])
                        row_rest = np.setdiff1d(rg, solution[r, :c])
                        avb1 = np.intersect1d(col_rest, row_rest)
                        sub_r, sub_c = r//3, c//3
                        avb2 = np.setdiff1d(np.arange(0, n+1), solution[sub_r*3:(sub_r+1)*3, sub_c*3:(sub_c+1)*3].ravel())
                        avb = np.intersect1d(avb1, avb2)
                        solution[r, c] = np.random.choice(avb, size=1)
                break
            except ValueError:
                pass
        start = solution.copy()
        start[np.random.choice([True, False], size=solution.shape, p=[self.mask_rate, 1 - self.mask_rate])] = 0
        return start, solution

    def reset(self, return_info=False):
        self.start, self.solution = self.generate_puzzle()
        self.observation = self.start.copy()
        self.attempt = 0
        info = self._get_info()
        if self.flatten:
            return (self.full_observation.flatten(), info) if return_info else self.full_observation.flatten()
        return (self.full_observation, info) if return_info else self.full_observation
    
    def print_render(self, observation=None):
        if observation is None:
            observation = self.observation
        print(f'\nattempt {self.attempt}')
        for i, (number_row, start_row) in enumerate(zip(observation, (self.start > 0).astype(bool))):
            if i in [3, 6]:
                print ('-------+-------+-------')
            print(' ', end='')
            for j, (number, start) in enumerate(zip(number_row, start_row)):
                if j in [3, 6]:
                    print('| ', end='')
                if number == 0:
                    number = ' '
                if start:
                    cprint(number, attrs=['underline'], end=' ')
                else:
                    print(number, end=' ')
            print()
    
    def step(self, action):
        if action is not None:
            changed = self.update_state(action)
        observation = self._get_obs(action)
        reward = self._get_reward(changed)
        done = reward > 0 or self.attempt == self.max_attempts
        info = self._get_info()
        if self.render:
            self.print_render()
        return observation, reward, done, info

    @classmethod
    def create_very_easy(cls, **kwargs):
        return cls(mask_rate=0.1)

    @classmethod
    def create_easy(cls, **kwargs):
        return cls(mask_rate=0.3)

    @classmethod
    def create_normal(cls, **kwargs):
        return cls(mask_rate=0.5)

    @classmethod
    def create_hard(cls, **kwargs):
        return cls(mask_rate=0.7)

    @classmethod
    def create_very_hard(cls, **kwargs):
        return cls(mask_rate=0.2)


class MinesweeperEnvBase(gym.Env):

    def __init__(self, height, width, bombs, render=False, flatten=True):
        self.height = height
        self.width = width
        self.bombs = bombs
        self.max_attempts = height * width
        self.render = render
        self.flatten = flatten

        if self.flatten:
            self.observation_space = spaces.MultiDiscrete([10] * height * width)
        else:
            self.observation_space = spaces.MultiDiscrete([[10] * height] * width)

        # x, y - coordinates on map puzzle
        self.action_space = spaces.MultiDiscrete([height, width])
    
    @property
    def trimmed_observation(self):
        return np.where(self.display.astype(bool), self.observation, 9)[1: -1, 1: -1]

    def _get_reward(self, hit):
        if hit == 'bomb':
            return -100
        elif hit == 'repeat':
            return -10
        elif hit == 'uncovered':
            return -1
        else:
            return 10

    def update_state(self, action):
        if self.display.sum() == 0:
            self.place_bombs(action)
        self.attempt += 1
        (x, y) = action
        x, y = x + 1, y + 1  # since there's a border around the obs
        repeat = self.last_action == (x, y)
        self.last_action = (x, y)
        if self.observation[x, y] == 9:
            return 'bomb'
        elif repeat:
            return 'repeat'
        elif self.display[x, y] == 0:
            self.uncover((x, y))
            return 'safe'
        return 'uncovered'

    def uncover(self, action):
        (x, y) = action
        if self.observation[action] == 0:
            surrounding = [(x - 1 + i, y - 1 + j) for i in range(3) for j in range(3)]
        else: 
            surrounding = [action]
        for a in surrounding:
            if a != action and self.observation[a] == 0 and self.display[a] == 0:
                self.display[a] = 1
                self.uncover(a)
            else:
                self.display[a] = 1

    def _get_obs(self):
        if self.flatten:
            return self.trimmed_observation.flatten()
        return self.trimmed_observation
    
    def _get_info(self):
        return {
            'attempt': self.attempt,
        }
    
    def place_bombs(self, action):
        (x, y) = action
        first_action_area = {(x - 1 + i, y - 1 + j) for i in range(3) for j in range(3)}
        bomb_coords = set()
        bomb_coords = bomb_coords | first_action_area
        # get bombs that aren't in the first action area
        while len(bomb_coords) < self.bombs + len(first_action_area):
            bomb_coords.add((np.random.randint(1, self.height), np.random.randint(1, self.width)))
        bomb_coords = bomb_coords - first_action_area
        # place bombs
        for (x, y) in bomb_coords:
            self.observation[x+1, y+1] = 9

        # count bombs
        for r in range(1, self.height):
            for c in range(1, self.width):
                if self.observation[r,c] != 9:
                    self.observation[r,c] = (self.observation[r-1: r+2, c-1: c+2] == 9).sum()
        
        # make border
        self.observation[0, :] = 1
        self.observation[self.height + 1, :] = 1
        self.observation[:, 0] = 1
        self.observation[:, self.width + 1] = 1

    def reset(self, return_info=False):
        self.last_action = (-1, -1)
        self.observation = np.zeros((self.height + 2, self.width + 2))
        self.display = np.zeros((self.height + 2, self.width + 2))
        self.attempt = 0
        info = self._get_info()
        if self.flatten:
            return (self.trimmed_observation.flatten(), info) if return_info else self.trimmed_observation.flatten()
        return (self.trimmed_observation, info) if return_info else self.trimmed_observation
    
    def print_render(self, observation=None):
        print(self.trimmed_observation)
    
    def step(self, action):
        if action is not None:
            hit = self.update_state(action)
        observation = self._get_obs()
        reward = self._get_reward(hit)
        done = hit == 'bomb' or self.attempt == self.max_attempts
        info = self._get_info()
        if self.render:
            self.print_render()
        return observation, reward, done, info


class MinesweeperEnvBeginner(MinesweeperEnvBase):
    def __init__(self, **kwargs):
        return super().__init__(height=8, width=8, bombs=10, **kwargs)


class MinesweeperEnvIntermediate(MinesweeperEnvBase):
    def __init__(self, **kwargs):
        return super().__init__(height=16, width=16, bombs=40, **kwargs)


class MinesweeperEnvExpert(MinesweeperEnvBase):
    def __init__(self, **kwargs):
        return super().__init__(height=16, width=30, bombs=99, **kwargs)