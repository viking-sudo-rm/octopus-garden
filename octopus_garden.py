from itertools import product, combinations
import random

term_values = [-1, 1]


def iter_worlds(d: int) -> iter:
    """A world is represented as a tuple over {-1, 1}."""
    return product(*[term_values for _ in range(d)])


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
            print("Found valid secret:", secret)
            valid_secrets.append(secret)
    return valid_secrets


def sample_semi_canonical(secret, d):
    world = random.choice(secret)
    unassigned_idx = random.randint(0, d - 1)
    return tuple(world[idx] if idx != unassigned_idx else 0 for idx in range(d))


def main(n=10, d=3):
    k = 2**(d - 1)  # Half the worlds should be included in the secret.
    worlds = list(iter_worlds(d))
    secret = random.choice(list(combinations(worlds, k)))
    print("True secret:", secret)

    ns = [5, 10, 50, 100, 1000, 2000, 5000, 10000]
    n_secrets = []

    for n in ns:
        utterances = [sample_semi_canonical(secret, d=3) for _ in range(n)]
        secrets = get_valid_secrets(utterances, worlds, k)
        n_secrets.append(len(secrets))
        print("=" * 10)
        print("# of secrets:", len(secrets))
        print(secrets)
    
    import matplotlib.pyplot as plt
    plt.plot(ns, n_secrets, marker=".")
    plt.xlabel("# utterances")
    plt.ylabel("# valid secrets")
    plt.xscale("log")
    plt.show()

if __name__ == "__main__":
    main()