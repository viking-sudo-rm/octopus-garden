from itertools import product, combinations
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.registry import Registry

experiments = Registry()

def iter_worlds(d: int) -> iter:
    """A world is represented as a tuple over {-1, 1}."""
    return product(*[[-1, 1] for _ in range(d)])


def iter_utterances(d: int) -> iter:
    return product(*[[-1, 0, 1] for _ in range(d)])


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


@experiments.register("universal-secrets")
def universal_secrets_of_size_k(args):
    worlds = list(iter_worlds(args.d))
    utterances = list(iter_utterances(args.d))
    results = np.zeros([2**args.d + 1, args.d + 1])
    pbar = tqdm(total=np.prod(results.shape))
    for k in range(0, 2**args.d + 1):
        for v in range(0, args.d + 1):
            for secret in combinations(worlds, k):
                valid_for_all = True
                for utterance in utterances:
                    if utterance.count(0) != v:
                        continue
                    if not any(evaluate(utterance, w) for w in secret):
                        valid_for_all = False
                        break
                if valid_for_all:
                    results[k, v] += 1
            pbar.update()
    pbar.close()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    im0 = axs[0].matshow(results.T)
    axs[0].set_title(f"Number of universal secrets")
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].matshow(results.T > 0)
    axs[1].set_title("Existence of universal secret")
    plt.colorbar(im1, ax=axs[1])
    plt.xlabel("Size of secret k")
    fig.text(0.125, 0.5, "Free variables v", va='center', rotation='vertical')
    axs[1].xaxis.set_ticks_position('bottom')
    # plt.ylabel("Free variables v")
    # plt.title(f"Universal secrets with d={args.d}")
    plt.tight_layout()
    plt.show()


@experiments.register("main")
def main(args):
    random.seed(args.seed)
    k = 2**(args.d - 2)  # Half the worlds should be included in the secret.
    worlds = list(iter_worlds(args.d))
    # TODO: Clean up 
    while True:
        secret = random.choice(list(combinations(worlds, k)))
        if all(not all(any(evaluate(u, w) for w in secret)
                   for u in iter_utterances(args.d) if u.count(0) == v) for v in range(1, args.d)):
            break
    print("True secret:", secret)
    print("=" * 10)

    ns = [5, 10, 50, 100, 1000, 10000, 100000]
    n_secrets = defaultdict(list)
    n_nu_secrets = defaultdict(list)

    canonical_utterances = secret  # With our representation, a world is a canonical utterance.
    secrets = get_valid_secrets(canonical_utterances, worlds, k)
    assert len(secrets) == 1

    # When v=d, all utterances have no info, so all secrets are universal. Therefore, loop to d - 1.
    for v in range(args.d):
        for n in ns:
            print("=" * 10, f"n={n}, v={v}", "=" * 10)
            utterances = [sample_semi_canonical_utterance(secret, args.d, v=v) for _ in range(n)] # OOPS
            utterances = list(set(utterances))
            print("First ten utterances:", secret_as_str(utterances[:10]))
            secrets = get_valid_secrets(utterances, worlds, k)

            nu_secrets = [s for s in secrets
                          if not all(any(evaluate(u, w) for w in s)
                          for u in iter_utterances(args.d) if u.count(0) == v)]
            n_secrets[v].append(len(secrets))
            n_nu_secrets[v].append(len(nu_secrets))

            print("# of all/non-universal secrets:", len(secrets), len(nu_secrets))
            for s in nu_secrets:
                if s == secret:
                    print("true:", s)
                else:
                    print("nu:", s)
    
    min_n_secrets = min(min(data) for data in n_secrets.values())

    import matplotlib.pyplot as plt
    for v in range(args.d):
        plt.plot(ns, n_secrets[v], marker=".", label=f"v={v}")
        plt.plot(ns, n_nu_secrets[v], marker=".", label=f"v={v}, nu")
    plt.axhline(y=min_n_secrets, linestyle="dashed", color="gray")
    plt.scatter([len(secret)], [1], color="black", label="canonical-only")
    plt.xlabel("# utterances")
    plt.ylabel("# valid secrets")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"# valid secrets in dataset with d={args.d} (best={min_n_secrets})")
    plt.legend()
    plt.show()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", choices=experiments.keys(), default="main")
    parser.add_argument("--check_zero", action="store_true", help="Check if >0 for universal secrets experiments.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    experiment_fn = experiments[args.experiment]
    experiment_fn(parse_args())