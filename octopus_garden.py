from itertools import product, combinations
import random
from collections import defaultdict
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_worlds(d: int) -> iter:
    """A world is represented as a tuple over {-1, 1}."""
    return product(*[[-1, 1] for _ in range(d)])


def evaluate(utterance, world) -> bool:
    """Return truth value of utterance in world.

    An utterance is a conjunction of d terms represented as a tuple in {-1, 0, 1}^d
    A world is a truth assignment to d bits represented as a tuple in {-1, 1}^d
    """
    return all(u * w >= 0 for u, w in zip(utterance, world))


def get_valid_secrets(utterances, worlds, k):
    valid_secrets = []
    for secret in combinations(worlds, k):
        if all(any(evaluate(u, w) for w in secret) for u in utterances):
            print("Found valid secret:", secret_as_str(secret))
            valid_secrets.append(secret)
    return valid_secrets


def sample_semi_canonical_utterance(secret, d: int, v: int = 1):
    world = random.choice(secret)
    idxs = list(range(d))
    unassigned_idxs = random.choice(list(combinations(idxs, d - v)))
    return tuple(world[idx] if idx in unassigned_idxs else 0 for idx in range(d))


symbols = [".", "+", "-"]

def secret_as_str(secret) -> str:
    return " ".join([utterance_as_str(u) for u in secret])

def utterance_as_str(utterance) -> str:
    return "".join([symbols[i] for i in utterance])

def main(args):
    random.seed(args.seed)
    k = 2**(args.d - 1)  # Half the worlds should be included in the secret.
    worlds = list(iter_worlds(args.d))
    secret = random.choice(list(combinations(worlds, k)))
    print("True secret:", secret)
    print("=" * 10)

    ns = [5, 10, 50, 100, 1000, 10000, 100000] #, 1000000]
    n_secrets = defaultdict(list)

    canonical_utterances = secret  # With our representation, a world is a canonical utterance.
    secrets = get_valid_secrets(canonical_utterances, worlds, k)
    assert len(secrets) == 1

    for v in range(args.d + 1):
        for n in ns:
            print("=" * 10, f"n={n}, v={v}", "=" * 10)
            utterances = [sample_semi_canonical_utterance(secret, args.d, v=v) for _ in range(n)] # OOPS
            print("First ten utterances:", secret_as_str(utterances[:10]))
            secrets = get_valid_secrets(utterances, worlds, k)
            n_secrets[v].append(len(secrets))
            print("# of secrets:", len(secrets))
    
    min_n_secrets = min(min(data) for data in n_secrets.values())

    import matplotlib.pyplot as plt
    for v in range(args.d + 1):
        plt.plot(ns, n_secrets[v], marker=".", label=f"v={v}")
    plt.axhline(y=min_n_secrets, linestyle="dashed", color="gray")
    plt.scatter([len(secret)], [1], color="black", label="canonical-only")
    plt.xlabel("# utterances")
    plt.ylabel("# valid secrets")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"# valid secrets in dataset with d={args.d} (best={min_n_secrets})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(parse_args())