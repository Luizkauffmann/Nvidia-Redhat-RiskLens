"""Warm the counterfactual cache (sequential, reverse, Shapley, macro variables) so tools answer instantly in a demo."""
import time

from risklens.cecl import attribution as A
from risklens.data.store import get_store


def main():
    s = get_store()
    prev, cur = s.dates[0], s.dates[-1]
    t0 = time.time()
    w = A.waterfall(prev, cur)
    A.waterfall(prev, cur, order=tuple(reversed(w["order"])))
    A.shapley_groups(prev, cur)
    A.macro_variable_shapley(prev, cur)
    print(f"attribution cache warm ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
