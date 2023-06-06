"""
Script de entreno para Pokémon Rojo Fuego de GameBoy Adance
"""
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

NUM_TEST_MATCHS = 1
NUM_K = 31
df = pd.DataFrame(columns= ['Time', 'HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'EnemyHP1', 'EnemyHP2', 'EnemyHP3', 'EnemyHP4', 'EnemyHP5',
                            'EnemyHP6', 'Reward', 'State', 'Won'], dtype = object)


def update_df(info, reward, state, won):
    df.loc[len(df.index)] = [dt.datetime.now(), info.get('HP1'), info.get("HP2"), info.get("HP3")
               , info.get("HP4"), info.get("HP5"), info.get("HP6"), info.get("EnemyHP1")
                , info.get("EnemyHP2"), info.get("EnemyHP3"), info.get("EnemyHP4")
                , info.get("EnemyHP5"), info.get("EnemyHP6"), reward, state, won]


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


game_states_debug = [
    'ENTRENADOR1',
    'ENTRENADOR2',
    'ENTRENADOR3',
    'ENTRENADOR4',
    'ENTRENADOR5'
]



#18 rival/36 SEGUNDO GYM/83 GIOVANNI/86 RIVAL/
game_states = [
    'NO_LEVELEO_GYM',
    'ENTRENADOR1',
    'ENTRENADOR2',
    'ENTRENADOR3',
    'ENTRENADOR4',
    'ENTRENADOR5',
    'ENTRENADOR6',
    'ENTRENADOR7',
    'ENTRENADOR8',
    'ENTRENADOR9',
    'ENTRENADOR10',
    'ENTRENADOR11',
    'ENTRENADOR12',
    'ENTRENADOR13',
    'ENTRENADOR14',
    'ENTRENADOR15',
    'ENTRENADOR16',
    'ENTRENADOR17',
    'ENTRENADOR18',
    'ENTRENADOR19',
    'ENTRENADOR20',
    'ENTRENADOR21',
    'ENTRENADOR22',
    'ENTRENADOR23',
    'ENTRENADOR24',
    'ENTRENADOR25',
    'ENTRENADOR26',
    'ENTRENADOR27',
    'ENTRENADOR28',
    'ENTRENADOR29',
    'ENTRENADOR30',
    'ENTRENADOR31',
    'ENTRENADOR32',
    'ENTRENADOR33',
    'ENTRENADOR34',
    'ENTRENADOR35',
    'ENTRENADOR36',
    'ENTRENADOR37',
    'ENTRENADOR38',
    'ENTRENADOR39',
    'ENTRENADOR40',
    'ENTRENADOR41',
    'ENTRENADOR42',
    'ENTRENADOR43',
    'ENTRENADOR44',
    'ENTRENADOR45',
    'ENTRENADOR46',
    'ENTRENADOR47',
    'ENTRENADOR48',
    'ENTRENADOR49',
    'ENTRENADOR50',
    'ENTRENADOR51',
    'ENTRENADOR52',
    'ENTRENADOR53',
    'ENTRENADOR54',
    'ENTRENADOR55',
    'ENTRENADOR56',
    'ENTRENADOR57',
    'TERCER_GYM',
    'ENTRENADOR58',
    'ENTRENADOR59',
    'ENTRENADOR60',
    'ENTRENADOR61',
    'ENTRENADOR62',
    'ENTRENADOR63',
    'ENTRENADOR64',
    'ENTRENADOR65',
    'ENTRENADOR66',
    'ENTRENADOR67',
    'ENTRENADOR68',
    'ENTRENADOR69',
    'ENTRENADOR70',
    'ENTRENADOR71',
    'ENTRENADOR72',
    'ENTRENADOR73',
    'ENTRENADOR74',
    'ENTRENADOR75',
    'ENTRENADOR76',
    'ENTRENADOR77',
    'ENTRENADOR78',
    'ENTRENADOR79',
    'ENTRENADOR80',
    'ENTRENADOR81',
    'ENTRENADOR82',
    'ENTRENADOR83',
    'ENTRENADOR84',
    'ENTRENADOR85',
    'CUARTO_GYM',
    'ENTRENADOR86',
    'ENTRENADOR87',
    'ENTRENADOR88',
    'ENTRENADOR89',
    'ENTRENADOR90',
    'ENTRENADOR91',
    'ENTRENADOR92',
    'ENTRENADOR93',
    'ENTRENADOR94',
    'ENTRENADOR95'

]

test_game_states = [
]

def test_model(args, num_matchs, state):
    game = ModelVsGame(args, need_display=False)

    won_matchs = 0
    total_rewards = 0
    for i in range(0, num_matchs):
        info, reward = game.play(False,False)
        if info.get('EnemyHP1') == 0 and info.get('EnemyHP2') == 0 and info.get('EnemyHP3') == 0 and info.get('EnemyHP4') == 0 and info.get('EnemyHP5') == 0 and info.get('EnemyHP6') == 0:
            won_matchs += 1
        total_rewards += reward
        update_df(info, reward, state ,won_matchs)
        # print(total_rewards)
        # print(info)

    return won_matchs, total_rewards


def main(argv):
    args = parse_cmdline(argv[1:])

    # Make a subset of all possible game states depending on args
    # i.e: first = 0, last = 5 would result in [0 1 2 3 4]
    game_states_subset = game_states[args.first:args.last]

    logger.log('================ Pokemon Rojo Fuego Trainer ================')
    logger.log('GAME STATES DISPONIBLES:')
    logger.log(game_states)

    # turn off verbose
    args.alg_verbose = True

    p1_model_path = args.load_p1_model



    #---------------------------------------------------------------------
    #TRAIN ON ALL BUT ONE INSTANCE, TEST ON THAT INSTANCE
    #AND DO THAT NUM_K TIMES


    if not args.test_only:
        #n = args.num_timesteps

        #Get the number of game states we have, python lysts start at 0 end at n-1
        num_states = len(game_states_subset)

        for i in range(NUM_K):
            for curr_test_state in range(num_states):

                #Extract the current test state
                test_game_states.append(game_states_subset.pop(curr_test_state))

                logger.log('CONJUNTO DE TEST:')
                logger.log(test_game_states)


                #Train
                for state in game_states_subset:
                    logger.log('ENTRENANDO EN EL ESTADO:%s' % (state))
                    args.state = state
                    args.load_p1_model = p1_model_path
                    old_path = p1_model_path + ".zip"
                    trainer = ModelTrainer(args)
                    p1_model_path = trainer.train()
                    if old_path != ".zip":
                        os.remove(old_path)
                        old_path = ''

                # Test performance of model on each test state
                logger.log('====== TESTEO MODELO ======')
                for state in test_game_states:
                    logger.log('TESTEANDO EN EL ESTADO:%s' % (state))
                    num_test_matchs = NUM_TEST_MATCHS
                    new_args = args
                    new_args.state = state
                    new_args.load_p1_model = p1_model_path
                    won_matchs, total_reward = test_model(new_args, num_test_matchs, new_args.state)
                    #percentage = won_matchs / num_test_matchs
                    logger.log('STATE:%s... COMBATES GANADOS:%d/%d TOTAL REWARDS:%d' % (
                    state, won_matchs, num_test_matchs, total_reward))


                #Put back the test state to the full game states list
                game_states_subset.insert(curr_test_state,test_game_states.pop())

        df.to_csv(args.output_basedir + 'outputDF.csv', encoding='utf-8', index=False)


    # ---------------------------------------------------------------------

    if args.play:
        args.state = 'ENTRENADOR96'
        args.load_p1_model = p1_model_path
        args.num_timesteps = 0

        player = ModelVsGame(args)
        #player = ModelTrainer(args)
        player.play(continuous=True, need_reset=False)
        #player.play(continuous=True)


if __name__ == '__main__':
    main(sys.argv)