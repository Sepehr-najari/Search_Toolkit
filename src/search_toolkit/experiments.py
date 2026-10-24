import json
from pathlib import Path

from search_toolkit.problems import GridProblem
from search_toolkit.algorithms import bfs, dfs, astar


def main():
    problem = GridProblem(
        width=8,
        height=6,
        start=(0, 0),
        goal=(7, 5),
        walls={
            (2, 0), (2, 1), (2, 2),
            (4, 2), (5, 2), (6, 2),
            (4, 3),
        },
    )

    results = {}

    for algorithm in (bfs, dfs, astar):
        result = algorithm(problem)

        results[algorithm.__name__] = {
            "found": result.found,
            "path": result.path,
            "cost": result.cost,
            "expanded": result.expanded,
            "generated": result.generated,
        }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    (output_dir / "grid_search.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()