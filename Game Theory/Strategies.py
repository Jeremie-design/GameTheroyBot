import numpy as np

def unconditional_cooperator(rnd, bot_h, opp_h):
    return 0

def unconditional_defector(rnd, bot_h, opp_h):
    return 1

def random_strategy(rnd, bot_h, opp_h):
    return int(np.random.random() < 0.5)


