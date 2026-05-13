"""
Author: Dr Zhibin Liao
Organisation: School of Computer Science and Information Technology, Adelaide University
Date: 12-Mar-2026
Description: This Python script lets you play the Flappy Bird game yourself.

The script is a part of Assignment 2 made for the course COMP 3027 Artificial Intelligence for the year
of 2026. Public distribution of this source code is strictly forbidden.
"""
from console import FlappyBirdEnv
from human_agent import HumanAgent
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=6)

    args = parser.parse_args()

    # Game environment
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=100)
    # a human agent (yourself) playing the game using keyboard
    human = HumanAgent(show_screen=True)

    while True:
        env.play(player=human)
        print('Game Over')
        if env.replay_game():
            print(f"Game restart.")
        else:
            break
