import numpy as np


# Always cooperate
def always_cooperate(rnd, bot_h, opp_h):
    return 0


# Always defect
def always_defect(rnd, bot_h, opp_h):
    return 1


# Randomly cooperate or defect
def random_strategy(rnd, bot_h, opp_h):
    return int(np.random.random() < 0.5)


# Copy the opponent's previous move
def copy_opponent(tnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    return opp_h[-1]


# Start by defecting, then copy the opponent
def suspicious_copy_opponent(rnd, bot_h, opp_h):
    if rnd == 1:
        return 1

    return opp_h[-1]


# Usually copy the opponent, but sometimes forgive defection
def forgiving_copy_opponent(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    opponent_last_move = opp_h[-1]

    if opponent_last_move == 0:
        return 0

    if np.random.random() < 0.20:
        return 0

    return 1


# Cooperate unless the opponent defected twice in a row
def forgive_two_defections(rnd, bot_h, opp_h):
    if rnd <= 2:
        return 0

    if opp_h[-1] == 1 and opp_h[-2] == 1:
        return 1

    return 0


# Cooperate until the opponent defects once,
# then defect forever
def cooperate_until_betrayed(rnd, bot_h, opp_h):
    if 1 in opp_h:
        return 1

    return 0


# Stay with a successful move, switch after failure
def win_stay_lose_shift(rnd, bot_h, opp_h):
    if rnd == 1:
        return 0

    bot_last = bot_h[-1]
    opponent_last = opp_h[-1]

    if bot_last == opponent_last:
        return bot_last

    return 1 - bot_last


# =========================================================
# STRATEGIES
# =========================================================

STRATEGIES = {

    "Always Cooperate (ALLC)": always_cooperate,

    "Always Defect (ALLD)": always_defect,

    "Random": random_strategy,

    "Tit-for-Tat (TFT)": copy_opponent,

    "Suspicious Tit-for-Tat (STFT)": suspicious_copy_opponent,

    "Generous Tit-for-Tat (GTFT)": forgiving_copy_opponent,

    "Tit-for-Two-Tats (TF2T)": forgive_two_defections,

    "Grim Trigger (GRIM)": cooperate_until_betrayed,

    "Pavlov / Win-Stay-Lose-Shift (WSLS)": win_stay_lose_shift,
}