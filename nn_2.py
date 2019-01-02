from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
import _thread as thread 

class SnakeNN:
    def __init__(self, initial_games = 100, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'model0/snake_nn_2.tflearn',n_snakes = 2, train_ind = 1):
        self.n_snakes = n_snakes
        self.train_ind = train_ind
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0],
                [[0, 1], 1],
                [[1, 0], 2],
                [[0, -1], 3]
                ]

    def initial_population(self,adv_model):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            _, prev_score, snakes, food = game.start(self.n_snakes)
            #print(snakes,self.train_ind)
            prev_observation = self.generate_observation(snakes[self.train_ind], food)
            prev_food_distance = self.get_food_distance(snakes[self.train_ind], food)
            for _ in range(self.goal_steps):
                game_actions = []
                for i in range(self.n_snakes):
                    if i == self.train_ind:
                        action, game_action = self.generate_action(snakes[i])
                        game_actions += [game_action]
                    else:
                        predictions = []
                        for action_i in range(-1, 2):
                            predictions.append(adv_model.predict(self.add_action_to_observation(prev_observation, action_i).reshape(-1, 5, 1)))
                        action_adv = np.argmax(np.array(predictions))        
                        game_actions += [self.get_game_action(snakes[i], action_adv - 1)]

                done, score, snakes, food  = game.step(game_actions)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    food_distance = self.get_food_distance(snakes[self.train_ind], food)
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snakes[self.train_ind], food)
                    prev_food_distance = food_distance
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        if(food == []):
            return np.array(snake[0])
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        print(X,y,len(X),len(y))
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model,adv_model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = SnakeGame()
            _, score, snakes, food = game.start(self.n_snakes)
            prev_observation = self.generate_observation(snakes[self.train_ind], food)
            for _ in range(self.goal_steps):
                game_actions = []
                for i in range(self.n_snakes):
                    predictions = []
                    for action in range(-1, 2):
                        if i == self.train_ind:
                            predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                            action = np.argmax(np.array(predictions))        
                            game_actions += [self.get_game_action(snakes[i], action - 1)]
                        else:
                            predictions.append(adv_model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))                        
                            action_i = np.argmax(np.array(predictions))        
                            game_actions += [self.get_game_action(snakes[i], action_i - 1)]
                done, score, snakes, food = game.step(game_actions)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snakes)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snakes[self.train_ind], food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, _, snakes, food = game.start(self.n_snakes)
        prev_observation = self.generate_observation(snakes[self.train_ind], food)
        for _ in range(self.goal_steps):
            thread.start_new_thread(game.check_input_thread, ())
            predictions = []
            for action in range(-1, 2):
               predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(predictions))
            game_action = self.get_game_action(snakes[self.train_ind], action - 1)
            done, _, snakes, food  = game.step([game.directions[0],game_action])
            if done:
                break
            else:
                prev_observation = self.generate_observation(snakes[self.train_ind], food)

    def train(self,adversary_filename = 'hungry/snake_nn_2.tflearn'):
        #load adversary model:
        adv_model = self.model()
        adv_model.load(adversary_filename)

        training_data = self.initial_population(adv_model)


        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        #load adversary model:
        #adv_model = self.model()
        #adv_model.load(self.adversary_filename)

        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    SnakeNN().train()
