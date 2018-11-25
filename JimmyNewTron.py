import pygame
import tflearn
import numpy
import math
from random import randint
from tensorflow import reset_default_graph
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
'''
    File name: JimmyNewTron.py
    Author: xenago (thanks to peterhogg https://github.com/peterhogg/tron)
    Python Version: 3.6
'''

'''
    HOW TO ADJUST GAME OPTIONS:
    
    - mode1/mode2: choose the control method for each player.
        keyboard        (use WASD for p1 and arrows for p2)
        random          (player moves randomly)
        ai_load_trained (loads model from disk and uses it to make movement decisions)
        ai_retrain      (same as ai_load_trained, but it also uses the games to retrain the model)
        ai_train_random (creates and saves new ai model, makes random decisions)
        
    - filename: choose target TFlearn data saved to disk (default: "TronNN.tflearn")
    
    - training_games: if one of the players selected ai_retrain or ai_train_random, this limits the number of automatic
                      training games which are played. Select 0 if you do not want it to automatically play games.
                      
    - speed: choose the speed the game's internal runs at. 10 is slow, 15 is normal, 50+ is recommended for training
    
'''

mode1 = "ai_load_trained"  # player 1
mode2 = "keyboard"  # player 2

filename = "TronNN.tflearn"

training_games = 100

speed = 15


# ----------------------------- #

# initialize the game engine
pygame.init()

# colour definitions
black = (0, 0, 0)
white = (200, 200, 200)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)

# sets up the window
x = 800
y = 800
size = [x, y]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Tron")

# sets the initial map
screen.fill(black)
for i in range(0, x, 20):
    pygame.draw.line(screen, white, [i, 0], [i, x], 1)
    pygame.draw.line(screen, white, [0, i], [y, i], 1)
pygame.display.flip()

# Variables for the first player
p1x = x / 4
p1y = y / 4
p1alive = True
p1colour = blue
p1score = 0

# Variables for the second player
p2x = (x * 3) / 4
p2y = (y * 3) / 4
p2alive = True
p2colour = yellow
p2score = 0

# Stores a bool as to whether or not the square has been traveled already
grid = [[False for temp in range(int(x / 20))] for temp in range(int(y / 20))]

# Sets up the loop for the game
done = False

# Sets the players initial directions
# 0 is right, 180 is left, 90 is up, 270 is down
p1direction = 0
p2direction = 180
p1_dir = 0
p2_dir = 0


# store training data from game
training_input_data = []
training_feedback_data = []

# Avoid retraining models
trained1 = False
trained2 = False

# train the AI using a single game first
first_run = True

# store winning player
pWin = 0


#
#   Return the game to its starting state
#
def reset():
    global screen, p1x, p1y, p1colour, p2x, p2y, p2colour, grid, p1alive, p2alive, trained1, trained2,\
        training_games, model1, model2, training_input_data, training_feedback_data, pWin, first_run
    screen.fill(black)
    for i in range(0, x, 20):
        pygame.draw.line(screen, white, [i, 0], [i, x], 1)
        pygame.draw.line(screen, white, [0, i], [y, i], 1)
    p1x = x / 4
    p1y = y / 4
    p1colour = blue
    p2x = (x * 3) / 4
    p2y = (y * 3) / 4
    p2colour = yellow
    grid = [[False for temp in range(int(x / 20))] for temp in range(int(y / 20))]
    pygame.draw.rect(screen, p1colour, [p1x + 1, p1y + 1, (x / 40) - 1, (x / 40) - 1])
    pygame.draw.rect(screen, p2colour, [p2x + 1, p2y + 1, (x / 40) - 1, (x / 40) - 1])
    pygame.display.flip()
    p1alive = True
    p2alive = True
    trained1 = False
    trained2 = False
    training_input_data = []
    training_feedback_data = []
    pWin = 0
    model1 = None
    model2 = None
    load_ai()
    first_run = False
    if training_games > 0:
        training_games = training_games - 1


#
#   Determine winner, update player scores
#
def update_score():
    global p1alive, p2alive, p1score, p2score, pWin
    # Update score
    if p1alive and not p2alive:
        p1score = p1score + 1
        pWin = 1
        print('\nPlayer 1 Score: ' + str(p1score))
        print('Player 2 Score: ' + str(p2score))
    if p2alive and not p1alive:
        pWin = 2
        p2score = p2score + 1
        print('\nPlayer 1 Score: ' + str(p1score))
        print('Player 2 Score: ' + str(p2score))


#
#   Loads NN model from disk
#
def load_model(player):
    global model1, model2
    if player == 1:
        model1.load(filename)
    else:
        model2.load(filename)


#
#   Determine left and right directions from current player heading
#
def left_right(player):
    if player == 1:
        if p1direction == 180:
            left = 270
            right = 90
        elif p1direction == 0:
            left = 90
            right = 270
        elif p1direction == 90:
            left = 180
            right = 0
        elif p1direction == 270:
            left = 0
            right = 180
    elif player == 2:
        if p2direction == 180:
            left = 270
            right = 90
        elif p2direction == 0:
            left = 90
            right = 270
        elif p2direction == 90:
            left = 180
            right = 0
        elif p2direction == 270:
            left = 0
            right = 180
    return left, right


#
#   Build a NN model
#
def create_model(player):
    global model1, model2
    network = input_data(shape=[None, 6, 1], name='input')
    network = fully_connected(network, 36, activation='relu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')
    if player == 1:
        model1 = model
    else:
        model2 = model


#
#   Update training data structures with latest game data
#
def update_training_data():
    global training_input_data, training_feedback_data, p1_dir, p2_dir
    p1_feedback = 0.5
    p2_feedback = 0.5
    if p1_ob_left == 1 and p1_dir == -1:
        p1_feedback = -1
    elif p1_ob_centre == 1 and p1_dir == 0:
        p1_feedback = -1
    elif p1_ob_right == 1 and p1_dir == 1:
        p1_feedback = -1
    if p2_ob_left == 1 and p2_dir == -1:
        p2_feedback = -1
    elif p2_ob_centre == 1 and p2_dir == 0:
        p2_feedback = -1
    elif p2_ob_right == 1 and p2_dir == 1:
        p2_feedback = -1
    p1_data = [p1_dir, p1_ob_left, p1_ob_centre, p1_ob_right, dist, angle1]
    training_input_data.append(p1_data)
    training_feedback_data.append(p1_feedback)
    p2_data = [p2_dir, p2_ob_left, p2_ob_centre, p2_ob_right, dist, angle2]
    training_input_data.append(p2_data)
    training_feedback_data.append(p2_feedback)


#
#   Determine distance between the players
#
def calc_distance():
    distance_raw = math.sqrt((p1x-p2x)**2 + (p1y-p2y)**2)
    max_dist = numpy.sqrt(x*x + y*y)
    distance = 1 - (distance_raw / max_dist)
    return distance


#
#   Determine angle between the players
#
def calc_angle(player):
    if player == 1:
        p1_angle_raw = math.atan2(p2y - p1y, p2x - p1x)
        if p1direction == 180:
            offset = 1
        if p1direction == 0:
            offset = 0
        if p1direction == 90:
            offset = 0.5
        if p1direction == 270:
            offset = 1.5
        p1_angle = (((p1_angle_raw / math.pi + 1) - offset) % 2) - 1
        return p1_angle
    else:
        p2_angle_raw = math.atan2(p1y - p2y, p1x - p2x)
        if p2direction == 180:
            offset = 1
        if p2direction == 0:
            offset = 0
        if p2direction == 90:
            offset = 0.5
        if p2direction == 270:
            offset = 1.5
        p2_angle = (((p2_angle_raw/math.pi + 1) - offset) % 2) - 1
        return p2_angle


#
#   Apply new movement direction to player based on pressed key
#
def handle_keyboard_input(key):
    global p1direction, p2direction, p1action, p2action, p1_dir, p2_dir
    print('Key pressed: ' + str(key))
    left, right = left_right(1)
    # Changes Player 1's direction based off the key the player pressed
    if not p1action:
        if key == pygame.K_a:
            if p1direction != 0:
                p1direction = 180
                p1action = True
        elif key == pygame.K_d:
            if p1direction != 180:
                p1direction = 0
                p1action = True
        elif key == pygame.K_w:
            if p1direction != 270:
                p1direction = 90
                p1action = True
        elif key == pygame.K_s:
            if p1direction != 90:
                p1direction = 270
                p1action = True
        if p1action:
            if p1direction == left:
                p1_dir = -1
            elif p1direction == right:
                p1_dir = 1
            else:
                p1_dir = 0
    # Changes Player 2's direction based off the key the player pressed
    left, right = left_right(2)
    if not p2action:
        if key == pygame.K_RIGHT:
            if p2direction != 180:
                p2direction = 0
                p2action = True
        elif key == pygame.K_UP:
            if p2direction != 270:
                p2direction = 90
                p2action = True
        elif key == pygame.K_DOWN:
            if p2direction != 90:
                p2direction = 270
                p2action = True
        elif key == pygame.K_LEFT:
            if p2direction != 0:
                p2direction = 180
                p2action = True
        if p2action:
            if p2direction == left:
                p2_dir = -1
            elif p2direction == right:
                p2_dir = 1
            else:
                p2_dir = 0


#
#   Exit the game
#
def game_quit():
    pygame.quit()


#
#   Prepare AI features
#
def load_ai():
    global model1, model2
    # handle loading ai
    reset_default_graph()
    if mode1 == "ai_load_trained" or mode1 == "ai_retrain" or mode1 == "ai_train_random":
        create_model(1)
    if mode2 == "ai_load_trained" or mode2 == "ai_retrain" or mode2 == "ai_train_random":
        create_model(2)
    if mode1 == "ai_load_trained" or mode1 == "ai_retrain":
        load_model(1)
    if mode2 == "ai_load_trained" or mode2 == "ai_retrain":
        load_model(2)
    if not first_run and mode1 == "ai_train_random":
        load_model(1)
    if not first_run and mode2 == "ai_train_random":
        load_model(2)


# Load AI features
load_ai()


# Attach game clock
clock = pygame.time.Clock()


# Main game loop
while not done:
    # Neither player has taken an action this frame
    p1action = False
    p2action = False

    # Redraw the players
    if p1alive or p2alive:
        pygame.draw.rect(screen, p1colour, [p1x + 1, p1y + 1, (x / 40) - 1, (x / 40) - 1])
        pygame.draw.rect(screen, p2colour, [p2x + 1, p2y + 1, (x / 40) - 1, (x / 40) - 1])
        pygame.display.flip()

    # Event handling
    for event in pygame.event.get():
        # Allows the loop to terminate when the user closes the window
        if event.type == pygame.QUIT:
            done = True

        # Handles keyboard input
        elif event.type == pygame.KEYDOWN:

            if mode1 == "keyboard" or mode2 == "keyboard":
                handle_keyboard_input(event.key)

            # Allows the user to reset the game if the space bar is hit and the game is over
            if event.key == pygame.K_SPACE and not p1alive and not p2alive:
                reset()
            # Allows quitting the game with a key
            elif event.key == pygame.K_ESCAPE:
                game_quit()

    if (not p1alive and not p2alive) and \
            (mode1 == "ai_retrain" or mode1 == "ai_train_random" or mode2 == "ai_retrain" or mode2 == "ai_train_random")\
            and training_games > 0:
        reset()

    # prepare game state data
    if p1alive or p2alive:
        dist = calc_distance()
        angle1 = calc_angle(1)
        angle2 = calc_angle(2)
        left1, right1 = left_right(1)
        # player 1left
        p1_ob_left = 0
        p1_temp_x = p1x
        p1_temp_y = p1y
        if left1 == 180:
            p1_temp_x -= 20
        elif left1 == 0:
            p1_temp_x += 20
        elif left1 == 90:
            p1_temp_y -= 20
        elif left1 == 270:
            p1_temp_y += 20
        if p1_temp_x >= 800 or p1_temp_x < 0 or p1_temp_y >= 800 or p1_temp_y < 0:
            p1_ob_left = 1
        # checks if player 1 will collide with another square
        try:
            if grid[int(p1_temp_x / 20) -1][int(p1_temp_y / 20)-1]:
                p1_ob_left = 1
        except:
            p1_ob_left = 1
        # player 1 centre
        p1_ob_centre = 0
        p1_temp_x = p1x
        p1_temp_y = p1y
        if p1direction == 180:
            p1_temp_x -= 20
        elif p1direction == 0:
            p1_temp_x += 20
        elif p1direction == 90:
            p1_temp_y -= 20
        elif p1direction == 270:
            p1_temp_y += 20
        if p1_temp_x >= 800 or p1_temp_x < 0 or p1_temp_y >= 800 or p1_temp_y < 0:
            p1_ob_centre = 1
        try:
            if grid[int(p1_temp_x / 20)-1][int(p1_temp_y / 20)-1]:
                p1_ob_centre = 1
        except:
            p1_ob_centre = 1
        # player 1 right
        p1_ob_right = 0
        p1_temp_x = p1x
        p1_temp_y = p1y
        if right1 == 180:
            p1_temp_x -= 20
        elif right1 == 0:
            p1_temp_x += 20
        elif right1 == 90:
            p1_temp_y -= 20
        elif right1 == 270:
            p1_temp_y += 20
        if p1_temp_x >= 800 or p1_temp_x < 0 or p1_temp_y >= 800 or p1_temp_y < 0:
            p1_ob_right = 1
        try:
            if grid[int(p1_temp_x / 20)-1][int(p1_temp_y / 20)-1]:
                p1_ob_right = 1
        except:
            p1_ob_right = 1
        left2, right2 = left_right(2)
        # player 2 left
        p2_ob_left = 0
        p2_temp_x = p1x
        p2_temp_y = p1y
        if left2 == 180:
            p2_temp_x -= 20
        elif left2 == 0:
            p2_temp_x += 20
        elif left2 == 90:
            p2_temp_y -= 20
        elif left2 == 270:
            p2_temp_y += 20
        if p2_temp_x >= 800 or p2_temp_x < 0 or p2_temp_y >= 800 or p2_temp_y < 0:
            p2_ob_left = 1
        try:
            if grid[int(p2_temp_x / 20)-1][int(p2_temp_y / 20)-1]:
                p2_ob_left = 1
        except:
            p2_ob_left = 1
        # player 2 centre
        p2_ob_centre = 0
        p2_temp_x = p1x
        p2_temp_y = p1y
        if p2direction == 180:
            p2_temp_x -= 20
        elif p2direction == 0:
            p2_temp_x += 20
        elif p2direction == 90:
            p2_temp_y -= 20
        elif p2direction == 270:
            p2_temp_y += 20
        if p2_temp_x >= 800 or p2_temp_x < 0 or p2_temp_y >= 800 or p2_temp_y < 0:
            p2_ob_centre = 1
        try:
            if grid[int(p2_temp_x / 20)-1][int(p2_temp_y / 20)-1]:
                p2_ob_centre = 1
        except:
            p2_ob_centre = 1
        # player 2 right
        p2_ob_right = 0
        p2_temp_x = p2x
        p2_temp_y = p2y
        if right2 == 180:
            p2_temp_x -= 20
        elif right2 == 0:
            p2_temp_x += 20
        elif right2 == 90:
            p2_temp_y -= 20
        elif right2 == 270:
            p2_temp_y += 20
        if p2_temp_x >= 800 or p2_temp_x < 0 or p2_temp_y >= 800 or p2_temp_y < 0:
            p2_ob_right = 1
        try:
            if grid[int(p2_temp_x / 20)-1][int(p2_temp_y / 20)-1]:
                p2_ob_right = 1
        except:
            p2_ob_right = 1

    # take AI action for player 1
    if p1alive and p2alive and (mode1 == "ai_load_trained" or mode1 == "ai_retrain"):
        probabilities = []
        X = numpy.array([-1, p1_ob_left, p1_ob_centre, p1_ob_right, dist, angle1]).reshape(-1, 6, 1)
        probabilities.append(model1.predict(X))
        X = numpy.array([0, p1_ob_left, p1_ob_centre, p1_ob_right, dist, angle1]).reshape(-1, 6, 1)
        probabilities.append(model1.predict(X))
        X = numpy.array([1, p1_ob_left, p1_ob_centre, p1_ob_right, dist, angle1]).reshape(-1, 6, 1)
        probabilities.append(model1.predict(X))
        action = numpy.argmax(numpy.array(probabilities))
        if action == 0:
            p1direction = left1
            p1_dir = 0
        elif action == 2:
            p1direction = right1
            p1_dir = 2
        else:
            p1_dir = 1

    # take AI action for player 2
    if p1alive and p2alive and (mode2 == "ai_load_trained" or mode2 == "ai_retrain"):
        probabilities = []
        X = numpy.array([-1, p2_ob_left, p2_ob_centre, p2_ob_right, dist, angle2]).reshape(-1, 6, 1)
        probabilities.append(model2.predict(X))
        X = numpy.array([0, p2_ob_left, p2_ob_centre, p2_ob_right, dist, angle2]).reshape(-1, 6, 1)
        probabilities.append(model2.predict(X))
        X = numpy.array([1, p2_ob_left, p2_ob_centre, p2_ob_right, dist, angle2]).reshape(-1, 6, 1)
        probabilities.append(model2.predict(X))
        action = numpy.argmax(numpy.array(probabilities))
        if action == 0:
            p2direction = left2
            p2_dir = -1
        elif action == 2:
            p2direction = right2
            p2_dir = 1
        else:
            p2_dir = 0

    # take random action for player 1
    if p1alive and p2alive and (mode1 == "ai_train_random" or mode1 == "random"):
        randMove = randint(0, 2)
        if randMove == 0:
            p1direction = left1
            p1_dir = -1
        elif randMove == 2:
            p2direction = right1
            p1_dir = 1
        else:
            p1_dir = 0
    # take random action for player 2
    if p1alive and p2alive and (mode2 == "ai_train_random" or mode2 == "random"):
        randMove = randint(0, 2)
        if randMove == 0:
            p2direction = left2
            p2_dir = -1
        elif randMove == 2:
            p2direction = right2
            p2_dir = 1
        else:
            p2_dir = 0

    # Update player 1 position
    if p1alive and p2alive:
        if p1direction == 180:
            p1x -= 20
        elif p1direction == 0:
            p1x += 20
        elif p1direction == 90:
            p1y -= 20
        elif p1direction == 270:
            p1y += 20
    # Update player 2 position
        if p2direction == 180:
            p2x -= 20
        elif p2direction == 0:
            p2x += 20
        elif p2direction == 90:
            p2y -= 20
        elif p2direction == 270:
            p2y += 20

    # checks if player 1 will travel off the map
    if p1x >= 800 or p1x < 0 or p1y >= 800 or p1y < 0:
        p1alive = False
        p1colour = red
    # checks if player 1 will collide with another square
    else:
        if grid[int(p1x / 20) - 1][int(p1y / 20) - 1]:
            p1alive = False
            p1colour = red
        # sets the square p1 is on to true
        grid[int(p1x / 20) - 1][int(p1y / 20) - 1] = True
    # checks if player 2 will travel off the map
    if p2x >= 800 or p2x < 0 or p2y >= 800 or p2y < 0:
        p2alive = False
        p2colour = red
    # checks if player 2 will collide with another square
    else:
        if grid[int(p2x / 20) - 1][int(p2y / 20) - 1]:
            p2alive = False
            p2colour = red
        # sets the square p1 is on to true
        grid[int(p2x / 20) - 1][int(p2y / 20) - 1] = True

    # Check for winner
    update_score()

    # Update AI training data
    if p1alive or p2alive:
        update_training_data()

    # Train AI
    if not p1alive:
        p1colour = red
        if not trained1 and (mode1 == "ai_retrain" or mode1 == "ai_train_random"):
            # feedback here
            update_training_data()
            X = numpy.array([training_input_data]).reshape(-1, 6, 1)
            y2 = numpy.array([training_feedback_data]).reshape(-1, 1)
            model1.fit(X, y2, n_epoch=2, shuffle=True, run_id=filename)
            model1.save(filename)
            # save
            trained1 = True
    if not p2alive:
        p2colour = red
        if not trained2 and (mode2 == "ai_retrain" or mode2 == "ai_train_random"):
            # feedback here
            update_training_data()
            X = numpy.array([training_input_data]).reshape(-1, 6, 1)
            y2 = numpy.array([training_feedback_data]).reshape(-1, 1)
            model2.fit(X, y2, n_epoch=2, shuffle=True, run_id=filename)
            model2.save(filename)
            # save
            trained2 = True

    # reset player direction
    p1_dir = 0
    p2_dir = 0

    # next frame
    clock.tick(speed)
game_quit()
