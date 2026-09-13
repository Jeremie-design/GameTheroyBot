# Game 
from LrBot import AdaptivePrisonerBot
from Strategies import STRATEGIES


# Settings
ROUNDS = 40000
LEARNING_RATE = 0.30
SEED = 42

# Standard Prisoner's Dilemma
# 0 = Cooperate, 1 = Defect
PAYOFFS = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (1, 1)
}


def run_simulation(opponent):
    bot = AdaptivePrisonerBot(
        lr=LEARNING_RATE,
        seed=SEED
    )

    bot_score = 0
    opponent_score = 0
    cooperations = 0
    defections = 0

    for rnd in range(1, ROUNDS + 1):
        opponent_move = opponent(
            rnd,
            bot.bot_history,
            bot.opp_history
        )

        bot_move = bot.act()

        if bot_move == 0:
            cooperations += 1
        else:
            defections += 1

        bot_payoff, opponent_payoff = PAYOFFS[
            (bot_move, opponent_move)
        ]

        bot.learn(
            bot_move,
            opponent_move,
            bot_payoff,
            max_reward=5
        )

        bot_score += bot_payoff
        opponent_score += opponent_payoff

    return {
        "bot_score": bot_score,
        "opponent_score": opponent_score,
        "bot_avg": bot_score / ROUNDS,
        "opponent_avg": opponent_score / ROUNDS,
        "cooperation_rate": cooperations / ROUNDS,
        "defection_rate": defections / ROUNDS
    }


def print_result(name, result):
    print(f"\n{name}")
    print("-" * 50)
    print(f"Bot score:          {result['bot_score']}")
    print(f"Opponent score:     {result['opponent_score']}")
    print(f"Bot average:        {result['bot_avg']:.4f}")
    print(f"Opponent average:   {result['opponent_avg']:.4f}")
    print(f"Cooperation:        {result['cooperation_rate'] * 100:.2f}%")
    print(f"Defection:          {result['defection_rate'] * 100:.2f}%")


if __name__ == "__main__":

    print("=" * 50)
    print("ADAPTIVE PRISONER'S DILEMMA")
    print("=" * 50)

    print(f"Rounds:        {ROUNDS}")
    print(f"Learning rate: {LEARNING_RATE}")

    results = {}

    for name, strategy in STRATEGIES.items():
        result = run_simulation(strategy)
        results[name] = result
        print_result(name, result)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Strategy':42}"
        f"{'Bot Avg':>10}"
        f"{'Opp Avg':>10}"
        f"{'Coop %':>10}"
        f"{'Defect %':>10}"
    )

    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:42}"
            f"{result['bot_avg']:10.4f}"
            f"{result['opponent_avg']:10.4f}"
            f"{result['cooperation_rate'] * 100:9.2f}%"
            f"{result['defection_rate'] * 100:9.2f}%"
        )

    print("\nSimulation complete.")
