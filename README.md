**What is Tron?**

A simple snake-style game where you avoid walls and any already-visited locations. 

**Quick Start:** At the top of the code, choose a mode for player 1 (blue) and 2 (yellow).

    - keyboard        (use WASD for p1 and arrows for p2)
    - random          (player moves randomly)
    - ai_load_trained (loads model from disk and uses it to make movement decisions)
    - ai_retrain      (same as ai_load_trained, but it also uses the games to retrain the model)
    - ai_train_random (creates and saves new ai model, makes random decisions)
    
**Additional Options:**
    
    - filename: choose target TFlearn data saved to disk (default: "TronNN.tflearn")
    - training_games: if one of the players selected ai_retrain or ai_train_random, this limits the number of automatic
                      training games which are played. Select 0 if you do not want it to automatically play games.
    - speed: choose the speed the game's internal runs at. 10 is slow, 15 is normal, 50+ is recommended for training

When the round is completed, press the space bar to replay.

*NOTE* This game requires pygame and tensorflow/TFLearn to run.
