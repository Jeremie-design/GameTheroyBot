import numpy as np


def always_cooperate(rnd, bot_h, opp_h):
    return 0


def always_defect(rnd, bot_h, opp_h):
    return 1


def random_strategy(rnd, bot_h, opp_h):
    return int(np.random.random() < 0.5)


def copy_opponent(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    return opp_h[-1]


def suspicious_copy_opponent(rnd, bot_h, opp_h):
    if rnd == 1:
        return 1

    return opp_h[-1]


def forgiving_copy_opponent(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    opponent_last_move = opp_h[-1]

    if opponent_last_move == 0:
        return 0

    if np.random.random() < 0.20:
        return 0

    return 1


def forgive_two_defections(rnd, bot_h, opp_h):
    if rnd <= 2:
        return 0

    if opp_h[-1] == 1 and opp_h[-2] == 1:
        return 1

    return 0


def cooperate_until_betrayed(rnd, bot_h, opp_h):
    if 1 in opp_h:
        return 1

    return 0


def win_stay_lose_shift(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    own_last = bot_h[-1]
    opponent_last = opp_h[-1]

    if opponent_last == 0:
        return own_last

    return 1 - own_last


STRATEGIES = {

    "Always Cooperate (ALLC)": always_cooperate,

    "Always Defect (ALLD)": always_defect,

    "Random": random_strategy,

    "Tit-for-Tat (TFT)": copy_opponent,

    "Suspicious Tit-for-Tat (STFT)": suspicious_copy_opponent,

    "Generous Tit-for-Tat (GTFT)": forgiving_copy_opponent,

    "Tit-for-Two-Tats (TF2T)": forgive_two_defections,

    "Nine-Move Trigger (GRIM-9)": cooperate_until_betrayed,

    "Pavlov / Win-Stay-Lose-Shift (WSLS)": win_stay_lose_shift,
}
