from itertools import product, combinations

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


def get_valid_secrets(utterances, d, k):
    valid_secrets = []
    worlds = list(iter_worlds(d))
    for secret in combinations(worlds, k):
        if all(any(evaluate(u, w) for w in secret) for u in utterances):
            print("Found valid secret:", secret)
            valid_secrets.append(secret)
    return valid_secrets


def main(d=3, k=2):
    utterances = [
        (1, 0, 0),
        (1, 0, -1),
    ]
    secrets = get_valid_secrets(utterances, d, k)
    print("=" * 10)
    print("# of secrets:", len(secrets))
    print(secrets)

if __name__ == "__main__":
    main()