import curses
from random import randint
import _thread as thread 
import time

class SnakeGame:
    def __init__(self, board_width = 20, board_height = 20, gui = False):
        self.score = 0
        self.done = False
        self.board = {'width': board_width, 'height': board_height}
        self.gui = gui
        self.snakes = []
        self.alive = []
        self.snakeCount = 0
        self.user_index = 0

        self.directions = []

    def start(self,n_snakes): # returns observations
        for _ in range(n_snakes):
            self.snake_init()
        self.generate_food()
        if self.gui: self.render_init()
        return self.generate_observations()

    def snake_init(self): # void
        x = randint(5, self.board["width"] - 5)
        y = randint(10, self.board["height"] - 5)
        snake = []                             # snake attribute: list of points
        vertical = randint(0,1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            snake.insert(0, point)
        self.snakes += [snake]
        self.snakeCount += 1
        self.alive += [True]
        self.directions += [0]

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
        for snake in self.snakes:
            if (food in snake): food = []
        self.food = food

    def render_init(self):
        curses.initscr()
        win = curses.newwin(self.board["width"] + 2, self.board["height"] + 2, 0, 0)
        curses.curs_set(0)
        win.nodelay(1)
        win.timeout(200)
        self.win = win
        self.render()

    def render(self):
        self.win.clear()
        self.win.border(0)
        self.win.addstr(0, 2, 'Score : ' + str(self.score) + ' ')
        self.win.addch(self.food[0], self.food[1], '@')
        for j in range(len(self.snakes)):
            if self.alive[j]:
                snake = self.snakes[j]
                for i, point in enumerate(snake):
                    if i == 0:
                        self.win.addch(point[0], point[1], '%')
                    else:
                        self.win.addch(point[0], point[1], '#')
        self.win.getch()

    def step(self, keys):
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        if self.done == True: self.end_game()
        # move each of the snakes:
        for i in range(self.snakeCount):
            key = keys[i]
            if self.alive[i]:
                if (abs(key - self.directions[i]) == 2):
                    key = self.directions[i] #cannot go backwards
                self.create_new_point(key,i)
                self.directions[i] = key
                if self.food_eaten(i):
                    if (i == 0):
                        self.score += 1
                    self.generate_food()
                else:
                    self.remove_last_point(i)
                self.check_collisions(i)
                if not self.alive[i]:
                    self.snakeCount -= 1
                    if i == 0:
                        self.done = True
                        print("You died")
                    else:
                        print("Enemy snake eliminated")
                        self.snakes[i] = []
        if(self.snakeCount < 2):
            self.done = True
        
        if self.gui: self.render()
        return self.generate_observations()

    def create_new_point(self, key,i):
        new_point = [self.snakes[i][0][0], self.snakes[i][0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snakes[i].insert(0, new_point)

    def remove_last_point(self,i):
        self.snakes[i].pop()

    def food_eaten(self,i):
        return self.snakes[i][0] == self.food

    def check_collisions(self,i):
        all_snakes = []
        for j in range(len(self.snakes)):
            if i == j:
                all_snakes += self.snakes[i][1:-1]
            else:
                all_snakes += self.snakes[j]
        corner0 = self.snakes[i][0][0] == 0
        corner1 = self.snakes[i][0][0] == self.board["width"] + 1
        corner2 = self.snakes[i][0][1] == 0 
        corner3 = self.snakes[i][0][1] == self.board["height"] + 1
        collision = self.snakes[i][0] in all_snakes
            #if i == 0:
            #    self.done = True
        if (corner0 or corner1 or corner2 or corner3 or collision):
            print("snake",i,": ",self.snakes[i], "has died; of collision?:",collision)
            self.alive[i] = False


    def generate_observations(self):
        return self.done, self.score, self.snakes, self.food

    def render_destroy(self):
        curses.endwin()

    def end_game(self):
        if self.gui: self.render_destroy()
        raise Exception("Game over")

    def check_input_thread(self): # run in new thread to monitor user input
        ss = curses.initscr()
        curses.cbreak()
        ss.keypad(1)
        key = ''
        try:
            key = ss.getch()
            ss.refresh()
            if key == curses.KEY_UP:
                key = 0
            elif key == curses.KEY_DOWN:
                key = 2
            elif key == curses.KEY_LEFT:
                key = 3
            elif key == curses.KEY_RIGHT:
                key = 1
            if not abs(key - self.directions[0]) == 2:
                self.directions[0] = key
        except:
            curses.endwin()

if __name__ == "__main__":
    game = SnakeGame(gui = True)
    game.start(2) #input number of snakes 
    
    #for _ in range(20):
    while not game.done:
        thread.start_new_thread(game.check_input_thread, ())
        #game.check_input_thread()
        game.step([game.directions[0],randint(0,3)])
        time.sleep(0.01)
