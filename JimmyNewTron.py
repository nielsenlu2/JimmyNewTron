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

    - mode: choose the control method for each player.
        keyboard        (use WASD for p1 and arrows for p2)
        random          (player moves randomly)
        ai_load_trained (loads model from disk and uses it to make movement decisions)
        ai_retrain      (same as ai_load_trained, but it also uses the games to retrain the model)
        ai_train_random (creates and saves new ai model, makes random decisions)

    - filename: choose target TFLearn data saved to disk (default: "TronNN.tflearn")

    - obstacles: add hazards in the game

    - training_games: if one of the players selected ai_retrain or ai_train_random, this limits the number of automatic
                      training games which are played. Select 0 if you do not want it to automatically play games.
                      
    - games_before_training: specify the number of games to play before training and flushing the data

    - speed: choose the refresh rate. 10 is slow, 15 is normal, 60 is recommended for training
    
'''

# ----------------------------- #
#            Options
# ----------------------------- #

p1_mode = "ai_retrain"
p2_mode = "random"

filename = "TronNN.tflearn"

num_obstacles = 15

training_games = 1000

games_before_training = 5

speed = 120

# ----------------------------- #

# colour definitions
black = (0, 0, 0)
white = (220, 220, 220)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
shaded_blue = (2, 2, 180)
yellow = (255, 255, 0)
shaded_yellow = (180, 180, 2)
obstacle = (140, 20, 240)

# Sets up the loop for the game
done = False

# initialize the game engine
pygame.init()

# sets up the window
x = 800
y = 800
size = [x, y]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("JimmyNewTron")

# train the AI using a single game first
first_run = True

# calculate size of NN
visible_block_range = 31  # how far the ai can 'see'
visible_blocks = (visible_block_range**2)-1
input_size = 8 + visible_blocks
connected_size = input_size * 2

# store winning player
pWin = 0
# count the number of frames
frame = 0
# count games played
game = 0

# players start with 0 score
p1_score = 0
p2_score = 0

# blocks in the way
obstacles = []

# store training data from game
training_input_data = []
training_feedback_data = []

# Attach game clock
clock = pygame.time.Clock()

# Initially model has not been trained (even if it has, this will build up some data before training)
trained = False


# build the grid-space to store information about game entities
# 0: nothing
# 1: player 1 location
# 2: player 2 location
# 3: player 1 traversed block
# 4: player 2 traversed block
# 5: obstacle
# 6: losing player
def build_grid():
    global grid
    grid = numpy.zeros((40, 40))
    grid[p1x][p1y] = 1
    grid[p2x][p2y] = 2
    generate_obstacles()


# Render the screen
def render_game():
    global frame
    # increase frame counter
    frame = frame + 1
    # clear display
    screen.fill(black)
    # draw grid
    for i in range(0, y, 20):
        pygame.draw.line(screen, white, [i, 0], [i, y], 1)
        pygame.draw.line(screen, white, [0, i], [y, i], 1)
    # draw boxes
    for index, value in numpy.ndenumerate(grid):
        if value == 1:  # PLAYER 1 LOCATION
            pygame.draw.rect(screen, blue, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
        elif value == 2:  # PLAYER 2 LOCATION
            pygame.draw.rect(screen, yellow, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
        elif value == 3:  # PLAYER 1 TRAVERSED
            pygame.draw.rect(screen, shaded_blue, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
        elif value == 4:  # PLAYER 2 TRAVERSED
            pygame.draw.rect(screen, shaded_yellow, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
        elif value == 5:  # OBSTACLE
            pygame.draw.rect(screen, obstacle, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
        elif value == 6:  # LOSING PLAYER
            pygame.draw.rect(screen, red, [index[0] * 20 + 1, index[1] * 20 + 1, (x / 40) - 1, (x / 40) - 1])
    # show frame
    pygame.display.flip()


# Build the list of obstacles, including game walls and random blocks
def generate_obstacles():
    global obstacles, grid
    # generate random blocks
    for i in range(num_obstacles):
        rx = randint(1, 38)
        ry = randint(1, 38)
        # make sure they're not in the way of the players
        while ((rx < (p1x + 3)) and (rx > (p1x - 3)) and (ry < (p1y + 3)) and (ry > (p1y - 3))) or \
                ((rx < (p2x + 3)) and (rx > (p2x - 3)) and (ry < (p2y + 3)) and (ry > (p2y - 3))):
            rx = randint(1, 38)
            ry = randint(1, 38)
        obstacles.append((rx, ry))
        # put them in the grid
        grid[rx][ry] = 5
    # generate walls
    for index, value in numpy.ndenumerate(grid):
        if index[0] in [0, 39] or index[1] in [0, 39]:
            grid[index[0]][index[1]] = 5


# Setup game session
def reset():
    global game, first_run, p1x, p1y, p1_alive, p1_score, p2x, p2y, p2_alive, p2_score, p1_dir, p2_dir, \
        grid, training_input_data, training_feedback_data, dist, p1_angle, p2_angle, p1_move_dir, p2_move_dir, \
        pWin, trained, frame
    game += 1
    print('\nGame: ' + str(game))
    if game > 1:
        first_run = False
    # Variables for the players
    p1x = 10
    p2x = 30
    p1y = 10
    p2y = 30
    p1_alive = True
    p2_alive = True
    p1_dir = 0  # 0 is right, 180 is left, 90 is up, 270 is down
    p2_dir = 180
    p1_move_dir = 0  # -1 is left, 0 is centre, 1 is right
    p2_move_dir = 0
    p1_angle = 1
    p2_angle = 1
    build_grid()
    pWin = 0
    dist = 1
    trained = False
    # count the number of frames
    frame = 0


# Determine left and right directions from current player heading
def left_right(player):
    if player == 1:
        direction = p1_dir
    else:
        direction = p2_dir
    if direction == 180:
        left = 270
        right = 90
    elif direction == 0:
        left = 90
        right = 270
    elif direction == 90:
        left = 180
        right = 0
    else:  # direction == 270:
        left = 0
        right = 180
    return left, right


# Apply new movement direction to player based on pressed key
def handle_keyboard_input(key):
    global p1_action, p2_action, p1_dir, p2_dir, p1_move_dir, p2_move_dir
    left, right = left_right(1)
    # Changes Player 1's direction based off the key the player pressed
    if p1_mode == "keyboard" and not p1_action and p1_alive:
        temp_dir = 0
        if key == pygame.K_a:
            if p1_dir != 0:
                temp_dir = 180
                p1_action = True
        elif key == pygame.K_d:
            if p1_dir != 180:
                temp_dir = 0
                p1_action = True
        elif key == pygame.K_w:
            if p1_dir != 270:
                temp_dir = 90
                p1_action = True
        elif key == pygame.K_s:
            if p1_dir != 90:
                temp_dir = 270
                p1_action = True
        if p1_action:
            if temp_dir == left:
                p1_move_dir = -1
            elif temp_dir == right:
                p1_move_dir = 1
            else:
                p1_move_dir = 0
    # Changes Player 2's direction based off the key the player pressed
    left, right = left_right(2)
    if p2_mode == "keyboard" and not p2_action and p2_alive:
        temp_dir = 0
        if key == pygame.K_RIGHT:
            if p2_dir != 180:
                temp_dir = 0
                p2_action = True
        elif key == pygame.K_UP:
            if p2_dir != 270:
                temp_dir = 90
                p2_action = True
        elif key == pygame.K_DOWN:
            if p2_dir != 90:
                temp_dir = 270
                p2_action = True
        elif key == pygame.K_LEFT:
            if p2_dir != 0:
                temp_dir = 180
                p2_action = True
        if p2_action:
            if temp_dir == left:
                p2_move_dir = -1
            elif temp_dir == right:
                p2_move_dir = 1
            else:
                p2_move_dir = 0


# Determine distance between the players
def calc_distance():
    global dist, prev_dist
    prev_dist = dist
    distance_raw = math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
    max_dist = 57
    dist = 1 - (distance_raw / max_dist)


# Determine angle between the players
def calc_angle(player):
    if player == 1:
        direction = p1_dir
        angle_raw = math.atan2(p2y - p1y, p2x - p1x)
    else:
        direction = p2_dir
        angle_raw = math.atan2(p1y - p2y, p1x - p2x)
    if direction == 180:
        offset = 1
    elif direction == 0:
        offset = 0
    elif direction == 90:
        offset = 0.5
    else:  # direction == 270:
        offset = 1.5
    angle = (((angle_raw / math.pi + 1) - offset) % 2) - 1
    return angle


# Build a neural network model
def create_model():
    global model
    reset_default_graph()
    network = input_data(shape=[None, input_size, 1], name='input')
    network = fully_connected(network, connected_size, activation='elu')
    network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adam', learning_rate=1e-2, loss='mean_square', name='target')
    model = tflearn.DNN(network, tensorboard_dir='log')


# Train neural network with data accumulated from last frames
def train_model():
    global training_input_data, training_feedback_data, filename
    if p1_mode in ["ai_retrain", "ai_train_random"] or p2_mode in ["ai_retrain", "ai_train_random"]:
        x_feedback = numpy.array([training_input_data]).reshape(-1, input_size, 1)
        y_feedback = numpy.array([training_feedback_data]).reshape(-1, 1)
        model.fit(x_feedback, y_feedback, n_epoch=3, shuffle=True, run_id=filename)
        # save
        model.save(filename)
        training_input_data = []
        training_feedback_data = []


# Update training data structures with data from this frame
def update_training_data():
    global training_input_data, training_feedback_data, p1_dir, p2_dir
    adj_frame = frame / 800
    frame_factor = adj_frame
    dist_factor = (dist - prev_dist)
    p1_feedback = 0.5 + frame_factor + dist_factor
    p2_feedback = p1_feedback
    if p1_move_dir == 0:
        p1_feedback += 0.05
    if p2_move_dir == 0:
        p2_feedback += 0.05
    if pWin == 1:
        p2_feedback = -1
        # apply penalty back in time if loss
        if len(training_feedback_data) >= 5:
            training_feedback_data[len(training_feedback_data) - 1] = -0.75
            training_feedback_data[len(training_feedback_data) - 3] = -0.50
    elif pWin == 2:
        p1_feedback = -1
        if len(training_feedback_data) >= 5:
            training_feedback_data[len(training_feedback_data) - 2] = -0.75
            training_feedback_data[len(training_feedback_data) - 4] = -0.50
    elif pWin == -1:
        p1_feedback = -1
        p2_feedback = -1
        if len(training_feedback_data) >= 5:
            training_feedback_data[len(training_feedback_data) - 2] = -0.75
            training_feedback_data[len(training_feedback_data) - 4] = -0.50
        if len(training_feedback_data) >= 5:
            training_feedback_data[len(training_feedback_data) - 1] = -0.75
            training_feedback_data[len(training_feedback_data) - 3] = -0.50

    # Player 1 training data
    training_input_data.append(p1_move_dir)
    training_input_data.append(ob[0][0])
    training_input_data.append(ob[0][1])
    training_input_data.append(ob[0][2])
    training_input_data.append(dist)
    training_input_data.append(p1_angle)
    training_input_data.append(dist-prev_dist)
    training_input_data.append(adj_frame)
    for index in range(visible_blocks):
        training_input_data.append(sur_blocks[0][index])
    # Player 1 feedback data
    training_feedback_data.append(p1_feedback)

    # Player 2 input data
    training_input_data.append(p2_move_dir)
    training_input_data.append(ob[1][0])
    training_input_data.append(ob[1][1])
    training_input_data.append(ob[1][2])
    training_input_data.append(dist)
    training_input_data.append(p2_angle)
    training_input_data.append(dist-prev_dist)
    training_input_data.append(adj_frame)
    for index in range(visible_blocks):
        training_input_data.append(sur_blocks[1][index])
    # Player 2 feedback data
    training_feedback_data.append(p2_feedback)

    # print('P1 Feedback: ' + str(p1_feedback))
    # print('P2 Feedback: ' + str(p2_feedback))


# Load model saved to disk
def load_model_from_disk():
    global model, filename
    # handle loading ai
    if p1_mode in ["ai_load_trained", "ai_retrain"] or p2_mode in ["ai_load_trained", "ai_retrain"]:
        model.load(filename)


# Move player according to predictions
def predict_move(player):
    global p1_dir, p1_move_dir, p2_dir, p2_move_dir
    probabilities = []
    if player == 1:
        if p1_mode in ['ai_load_trained', 'ai_retrain', 'ai_train_random'] and frame == 1:
            action = randint(0, 2)
            print("RANDOM ACTION TAKEN")
            if action == 0:
                p1_move_dir = -1
            elif action == 2:
                p1_move_dir = 1
            else:
                p1_move_dir = 0
        else:
            adj_frame = frame / 800
            x_input_temp = [-1, ob[0][0], ob[0][1], ob[0][2], dist, p1_angle, dist-prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[0][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            x_input_temp = [0, ob[0][0], ob[0][1], ob[0][2], dist, p1_angle, dist - prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[0][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            x_input_temp = [1, ob[0][0], ob[0][1], ob[0][2], dist, p1_angle, dist - prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[0][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            action = numpy.argmax(numpy.array(probabilities))
            if numpy.max(numpy.array(probabilities)) < 0:
                action = randint(0, 2)
                print("RANDOM ACTION TAKEN")
            if probabilities[0] == probabilities[1] == probabilities[2]:
                p1_move_dir = 0
            elif action == 0:
                p1_move_dir = -1
            elif action == 2:
                p1_move_dir = 1
            else:
                p1_move_dir = 0
    elif player == 2:
        if p1_mode in ['ai_load_trained', 'ai_retrain', 'ai_train_random'] and frame == 1:
            action = randint(0, 2)
            print("RANDOM ACTION TAKEN")
            if action == 0:
                p2_move_dir = -1
            elif action == 2:
                p2_move_dir = 1
            else:
                p2_move_dir = 0
        else:
            adj_frame = frame / 800
            x_input_temp = [-1, ob[1][0], ob[1][1], ob[1][2], dist, p2_angle, dist-prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[1][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            x_input_temp = [0, ob[1][0], ob[1][1], ob[1][2], dist, p2_angle, dist - prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[1][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            x_input_temp = [1, ob[1][0], ob[1][1], ob[1][2], dist, p2_angle, dist - prev_dist, adj_frame]
            for index in range(visible_blocks):
                x_input_temp.append(sur_blocks[1][index])
            x_input = numpy.array([x_input_temp]).reshape(-1, input_size, 1)
            probabilities.append(model.predict(x_input))
            action = numpy.argmax(numpy.array(probabilities))
            if numpy.max(numpy.array(probabilities)) < 0:
                action = randint(0, 2)
                print("RANDOM ACTION TAKEN")
            if probabilities[0] == probabilities[1] == probabilities[2]:
                p2_move_dir = 0
            if action == 0:
                p2_move_dir = -1
            elif action == 2:
                p2_move_dir = 1
            else:
                p2_move_dir = 0


# Move player randomly, with a certain number of retries
def random_move(player):
    global p1_dir, p1_move_dir, p2_dir, p2_move_dir
    retries = 10
    if player == 1:
        i = retries
        good = False
        while i > 0 and not good:
            rand_move = randint(0, 2)
            if rand_move == 0 and ob[0][0] == 0:
                p1_move_dir = -1
                good = True
            elif rand_move == 2 and ob[0][2] == 0:
                p1_move_dir = 1
                good = True
            elif rand_move == 1 and ob[0][1] == 0:
                p1_move_dir = 0
                good = True
            i -= 1
    elif player == 2:
        i = retries
        good = False
        while i > 0 and not good:
            rand_move = randint(0, 2)
            if rand_move == 0 and ob[1][0] == 0:
                p2_move_dir = -1
                good = True
            elif rand_move == 2 and ob[1][2] == 0:
                p2_move_dir = 1
                good = True
            elif rand_move == 1 and ob[1][1] == 0:
                p2_move_dir = 0
                good = True
            i -= 1


# move players according to p_dir and p_move_dir
def handle_player_movement():
    global p1x, p2x, p1y, p2y, grid, p1_dir, p2_dir, p1_move_dir, p2_move_dir, p1x_prev, p2x_prev, p1y_prev, p2y_prev
    if grid[p1x][p1y] != 6 and grid[p2x][p2y] != 6:
        p1x_prev = p1x
        p2x_prev = p2x
        p1y_prev = p1y
        p2y_prev = p2y
        # set current position
        grid[p1x][p1y] = 3
        if p1_move_dir == -1:
            if p1_dir == 0:
                p1y -= 1
            elif p1_dir == 90:
                p1x -= 1
            elif p1_dir == 180:
                p1y += 1
            else:  # p1_dir == 270:
                p1x += 1
        elif p1_move_dir == 0:
            if p1_dir == 0:
                p1x += 1
            elif p1_dir == 90:
                p1y -= 1
            elif p1_dir == 180:
                p1x -= 1
            else:  # p1_dir == 270:
                p1y += 1
        else:  # p1_move_dir == 1:
            if p1_dir == 0:
                p1y += 1
            elif p1_dir == 90:
                p1x += 1
            elif p1_dir == 180:
                p1y -= 1
            else:  # p1_dir == 270:
                p1x -= 1
        # set new position
        grid[p1x][p1y] = 1
        left, right = left_right(1)
        if p1_move_dir == -1:
            p1_dir = left
        elif p1_move_dir == 1:
            p1_dir = right
        # set current position
        grid[p2x][p2y] = 4
        if p2_move_dir == -1:
            if p2_dir == 0:
                p2y -= 1
            elif p2_dir == 90:
                p2x -= 1
            elif p2_dir == 180:
                p2y += 1
            else:  # p2_dir == 270:
                p2x += 1
        elif p2_move_dir == 0:
            if p2_dir == 0:
                p2x += 1
            elif p2_dir == 90:
                p2y -= 1
            elif p2_dir == 180:
                p2x -= 1
            else:  # p2_dir == 270:
                p2y += 1
        else:  # p2_move_dir == 1:
            if p2_dir == 0:
                p2y += 1
            elif p2_dir == 90:
                p2x += 1
            elif p2_dir == 180:
                p2y -= 1
            else:  # p2_dir == 270:
                p2x -= 1
        # set new position
        grid[p2x][p2y] = 2
        left, right = left_right(2)
        if p2_move_dir == -1:
            p2_dir = left
        elif p2_move_dir == 1:
            p2_dir = right


#
#   Determine if any player has won.
#
def determine_score():
    global pWin, p1_alive, p2_alive, p1_score, p2_score, p1x_prev, p1y_prev, p2x_prev, p2y_prev
    p1_win = 0
    p2_win = 0
    if p1_move_dir == -1 and ob[0][0] == 1:
        p1_win = 2
        p1_alive = False
    elif p1_move_dir == 0 and ob[0][1] == 1:
        p1_win = 2
        p1_alive = False
    elif p1_move_dir == 1 and ob[0][2] == 1:
        p1_win = 2
        p1_alive = False
    if p2_move_dir == -1 and ob[1][0] == 1:
        p2_win = 1
        p2_alive = False
    elif p2_move_dir == 0 and ob[1][1] == 1:
        p2_win = 1
        p2_alive = False
    elif p2_move_dir == 1 and ob[1][2] == 1:
        p2_win = 1
        p2_alive = False
    if (p2x == p1x and p2y == p1y) or (p2x == p1x_prev and p2y == p1y_prev) or (p2x_prev == p1x and p2y_prev == p1y) or\
            (p2x_prev == p1x_prev and p2y_prev == p1y_prev):
        p1_win = 2
        p2_win = 1
    # select random winner if not training AI
    if p1_win == 2 and p2_win == 1:
        if p1_mode in ['ai_retrain', 'ai_train_random'] or p2_mode in ['ai_retrain', 'ai_train_random']:
            pWin = -1
            p2_win = 0
            p1_win = 0
            grid[p1x][p1y] = 6
            grid[p2x][p2y] = 6
        else:
            pWin = randint(1, 2)
            if pWin == 1:
                p2_win = 1
                p1_win = 0
            else:
                p1_win = 2
                p2_win = 0
    if p2_win == 1:
        grid[p2x][p2y] = 6
        pWin = 1
        p1_score += 1
        p2_alive = False
    if p1_win == 2:
        grid[p1x][p1y] = 6
        pWin = 2
        p2_score += 1
    if p1_win == 2 or p2_win == 1:
        if p1_win == 2 and p2_win != 1:
            print('Winner: P2')
        elif p1_win != 2 and p2_win == 1:
            print('Winner: P1')
        print('P1 Score: ' + str(p1_score))
        print('P2 Score: ' + str(p2_score))


# Game state
def analyze_game_state():
    global dist, p1_angle, p2_angle, ob, p1_dir, p2_dir, sur_blocks, visible_block_range
    calc_distance()
    p1_angle = calc_angle(1)
    p2_angle = calc_angle(2)
    left1, right1 = left_right(1)
    # player 1 left
    ob[0][0] = 0
    p1_temp_x = p1x
    p1_temp_y = p1y
    if left1 == 180:
        p1_temp_x -= 1
    elif left1 == 0:
        p1_temp_x += 1
    elif left1 == 90:
        p1_temp_y -= 1
    elif left1 == 270:
        p1_temp_y += 1
    if grid[p1_temp_x][p1_temp_y] > 0:
        ob[0][0] = 1
    # player 1 centre
    ob[0][1] = 0
    p1_temp_x = p1x
    p1_temp_y = p1y
    if p1_dir == 180:
        p1_temp_x -= 1
    elif p1_dir == 0:
        p1_temp_x += 1
    elif p1_dir == 90:
        p1_temp_y -= 1
    elif p1_dir == 270:
        p1_temp_y += 1
    if grid[p1_temp_x][p1_temp_y] > 0:
        ob[0][1] = 1
    # player 1 right
    ob[0][2] = 0
    p1_temp_x = p1x
    p1_temp_y = p1y
    if right1 == 180:
        p1_temp_x -= 1
    elif right1 == 0:
        p1_temp_x += 1
    elif right1 == 90:
        p1_temp_y -= 1
    elif right1 == 270:
        p1_temp_y += 1
    if grid[p1_temp_x][p1_temp_y] > 0:
        ob[0][2] = 1
    left2, right2 = left_right(2)
    # player 2 left
    ob[1][0] = 0
    p2_temp_x = p2x
    p2_temp_y = p2y
    if left2 == 180:
        p2_temp_x -= 1
    elif left2 == 0:
        p2_temp_x += 1
    elif left2 == 90:
        p2_temp_y -= 1
    elif left2 == 270:
        p2_temp_y += 1
    if grid[p2_temp_x][p2_temp_y] > 0:
        ob[1][0] = 1
    # player 2 centre
    ob[1][1] = 0
    p2_temp_x = p2x
    p2_temp_y = p2y
    if p2_dir == 180:
        p2_temp_x -= 1
    elif p2_dir == 0:
        p2_temp_x += 1
    elif p2_dir == 90:
        p2_temp_y -= 1
    elif p2_dir == 270:
        p2_temp_y += 1
    if grid[p2_temp_x][p2_temp_y] > 0:
        ob[1][1] = 1
    # player 2 right
    ob[1][2] = 0
    p2_temp_x = p2x
    p2_temp_y = p2y
    if right2 == 180:
        p2_temp_x -= 1
    elif right2 == 0:
        p2_temp_x += 1
    elif right2 == 90:
        p2_temp_y -= 1
    elif right2 == 270:
        p2_temp_y += 1
    if grid[p2_temp_x][p2_temp_y] > 0:
        ob[1][2] = 1
        ob[1][2] = 1
    sur_blocks = numpy.zeros((2, visible_block_range ** 2 - 1))
    get_surroundings()
    # print()


# build information about blocks surrounding player
# starting from immediate left, going clockwise, capture state of surrounding blocks
def get_surroundings():
    global visible_block_range, sur_blocks
    centre = (visible_block_range - 1) / 2
    for p_num in range(2):
        for i in range(visible_block_range):
            for j in range(visible_block_range):
                if i != centre or j != centre:  # the centre is the player so is obviously occupied
                    if p_num == 0:
                        temp_x = p1x
                        temp_y = p1y
                    else:  # p_num == 1
                        temp_x = p2x
                        temp_y = p2y
                    if i > centre:  # right
                        temp_x += i-centre
                    elif i < centre:  # left
                        temp_x -= centre-i
                    if j > centre:  # down
                        temp_y += j-centre
                    elif j < centre:  # up
                        temp_y -= centre-j
                    try:
                        if grid[temp_x][temp_y] > 0:
                            sur_blocks[p_num][i] = 1
                    except:
                        sur_blocks[p_num][i] = 1


#
#   Exit the game
#
def game_quit():
    pygame.quit()


# Build AI Model if needed
if p1_mode in ['ai_load_trained', 'ai_retrain', 'ai_train_random'] or \
        p2_mode in ['ai_load_trained', 'ai_retrain', 'ai_train_random']:
    create_model()

# Load model from disk if needed
load_model_from_disk()

# Start the game
reset()

# Main game loop
while not done:
    # Render frame
    render_game()

    # Neither player has taken an action this frame
    p1_action = False
    p2_action = False

    # reset movement info
    # save obstructions
    ob = numpy.zeros((2, 3))
    ob[0][0] = 0  # Player 1
    ob[0][1] = 0
    ob[0][2] = 0
    ob[1][0] = 0  # Player 2
    ob[1][1] = 0
    ob[1][2] = 0
    p1_move_dir = 0
    p2_move_dir = 0

    # Event handling
    for event in pygame.event.get():
        # Allows the loop to terminate when the user closes the window
        if event.type == pygame.QUIT:
            done = True

        # Handles keyboard input
        elif event.type == pygame.KEYDOWN:

            if p1_mode == "keyboard" or p2_mode == "keyboard":
                handle_keyboard_input(event.key)

            # Allows the user to reset the game if the space bar is hit and the game is over
            if event.key == pygame.K_SPACE and (not p1_alive or not p2_alive):
                reset()
            # Allows quitting the game with a key
            elif event.key == pygame.K_ESCAPE:
                game_quit()

    # reset the game if in training mode
    if (not p1_alive or not p2_alive) and \
            (p1_mode in ["ai_retrain", "ai_train_random"] or p2_mode in ["ai_retrain", "ai_train_random"]) \
            and training_games > 0:
        training_games -= 1
        reset()

    if p1_alive and p2_alive:
        # save game state
        analyze_game_state()
        if p1_mode in ["ai_retrain", "ai_load_trained"]:
            predict_move(1)
        elif p1_mode in ["random", "ai_train_random"]:
            random_move(1)
        if p2_mode in ["ai_retrain", "ai_load_trained"]:
            predict_move(2)
        elif p2_mode in ["random", "ai_train_random"]:
            random_move(2)
        # handle player movement
        handle_player_movement()
        # score
        determine_score()
        # update training data
        if p1_mode in ["ai_retrain", "ai_train_random"] or p2_mode in ["ai_retrain", "ai_train_random"]:
            update_training_data()
        # train network every 25 games
        if game % games_before_training == 0 and not trained:
            train_model()
            trained = True
        if not p1_alive or not p2_alive:
            p1_alive = False
            p2_alive = False

    # next frame
    clock.tick(speed)
game_quit()
