import retro
import os
import shutil
# hack to disable flood of warnings
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import retro
import datetime
import joblib
import argparse
import logging
import numpy as np
from stable_baselines import logger
import random
import pandas as pd
import datetime as dt

sys.path.append('.')

from model_trainer import ModelTrainer
from model_vs_game import ModelVsGame
"""
EJEMPLO AIRSTRIKER FUNCIONAL
env = retro.make(game='Airstriker-Genesis', record='.')
env.reset()

done = False

while not done:
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
        env.close()
"""

def parse_cmdline(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='ppo2')
    parser.add_argument('--nn', type=str, default='CnnPolicy')
    parser.add_argument('--model_desc', type=str, default='CNN')
    parser.add_argument('--env', type=str, default='Pokemon_Rojo_Fuego')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--num_players', type=int, default='1')
    parser.add_argument('--num_env', type=int, default=6)
    parser.add_argument('--num_timesteps', type=int, default=3000000)
    parser.add_argument('--output_basedir', type=str, default='~/OUTPUT')
    parser.add_argument('--load_p1_model', type=str, default='')
    parser.add_argument('--alg_verbose', default=True, action='store_false')
    parser.add_argument('--info_verbose', default=True, action='store_false')
    parser.add_argument('--display_width', type=int, default='1440')
    parser.add_argument('--display_height', type=int, default='810')
    parser.add_argument('--deterministic', default=False, action='store_true')
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--last', type=int, default=4)
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--play', default=False, action='store_true')

    args = parser.parse_args(argv)

    if args.info_verbose is False:
        logger.set_level(logger.DISABLED)

    logger.log("=========== Parametros ===========")
    logger.log(argv[1:])

    return args



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))




def main():


        print("Pokemon_Rojo_Fuego" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("Pokemon_Rojo_Fuego", inttype=retro.data.Integrations.ALL)
        obs = env.reset()
        print(env)

        while True:
            #env.render()
            obs, rew, done, info = env.step(env.action_space.sample())

            #print(obs)

            if rew != 0:
                print("Reward: "+ str(rew))

            if done:
                print("Done: "+ str(done))
            env.render()
            if done:
                    obs = env.reset()
        env.close()


if __name__ == "__main__":
        main()

