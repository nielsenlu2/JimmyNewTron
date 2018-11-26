**What is Tron?**

Tron is a simple 2-player snake-style game where players try to avoid walls and any already-visited locations. JimmyNewTron is a version which has been rewritten and supports training a neural network model to serve as a non-player character. I have included a model which has been trained for about 1000 frames.

**Quick Start:** At the top of the code, choose a mode for player 1 (blue) and 2 (yellow).

    - keyboard        - use WASD for p1 and arrows for p2
    - random          - player moves randomly
    - ai_load_trained - load model from disk, use it to make movement decisions
    - ai_retrain      - load model from disk, use it to make movement decisions, train model based on game data
    - ai_train_random - create a new model, move player randomly, train model based on game data
                        note: this will overwrite your existing model
    
**Additional Options:** These can be changed if desired
    
    - filename              - choose target for TFlearn data on disk (default: "TronNN.tflearn")
    - obstacles             - choose the number of random hazards in the game (0+)   
    - training_games        - if one of the players selected ai_retrain or ai_train_random, this limits the number of automatic
                              training games which are played. Select 0 if you do not want it to automatically play games.
    - games_before_training - specify the number of games to play before training and flushing the data
    - speed                 - choose the refresh rate. 10 is slow, 15 is normal, 60 is recommended for training

When the round is completed, press the space bar to replay.

*NOTE* This game requires pygame and tensorflow/TFLearn to run.
