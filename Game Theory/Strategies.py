import numpy as np

def unconditional_cooperator(rnd, bot_h, opp_h):
    return 0

def unconditional_defector(rnd, bot_h, opp_h):
    return 1

def random_strategy(rnd, bot_h, opp_h):
    return int(np.random.random() < 0.5)


def tit_for_tat(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0
    return opp_h[-1]


def suspicious_tit_for_tat(rnd, bot_h, opp_h):
    if rnd == 1:
        return 1
    return opp_h[-1]

def generous_tit_for_tat(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0
    opponent_last_move = opp_h[-1]

    if opponent_last_move == 0:
        return 0

    if np.random.random() < 0.20:
        return 0
    return 1

def tit_for_two_tats(rnd, bot_h, opp_h):
   
    if rnd <= 2:
        return 0

    if opp_h[-1] == 1 and opp_h[-2] == 1:
        return 1

    return 0



def grim_trigger(rnd, bot_h, opp_h):

    if 1 in opp_h:
        return 1

    return 0


def pavlov(rnd, bot_h, opp_h):
   
    if rnd == 1:
        return 0

    bot_last = bot_h[-1]
    opponent_last = opp_h[-1]

    if bot_last == opponent_last:
        return bot_last

    return 1 - bot_last

STRATEGIES = {

    "Unconditional Cooperator (ALLC)": unconditional_cooperator,

    "Unconditional Defector (ALLD)": unconditional_defector,

    "Random": random_strategy,

    "Tit-for-Tat (TFT)": tit_for_tat,

    "Suspicious Tit-for-Tat (STFT)": suspicious_tit_for_tat,

    "Generous Tit-for-Tat (GTFT)": generous_tit_for_tat,

    "Tit-for-Two-Tats (TF2T)": tit_for_two_tats,

    "Grim Trigger (GRIM)": grim_trigger,

    "Pavlov / Win-Stay-Lose-Shift (WSLS)": pavlov,
}