from typing import Protocol, TypeVar

State = TypeVar("State")


class SearchProblem(Protocol[State]):
    def initial_state(self) -> State:
        ...

    def is_goal(self, state: State) -> bool:
        ...

    def neighbors(self, state: State) -> list[State]:
        ...

    def step_cost(self, state: State, neighbor: State) -> float:
        ...

    def heuristic(self, state: State) -> float:
        ...


class GridProblem:
    def __init__(
        self,
        width: int,
        height: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        walls: set[tuple[int, int]] | None = None,
    ):
        if width < 1 or height < 1:
            raise ValueError("Grid dimensions must be positive")

        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.walls = walls or set()

        if not self._in_bounds(start) or not self._in_bounds(goal):
            raise ValueError("Start and goal must be inside the grid")

        if start in self.walls or goal in self.walls:
            raise ValueError("Start and goal cannot be walls")

    def _in_bounds(self, state: tuple[int, int]) -> bool:
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def initial_state(self) -> tuple[int, int]:
        return self.start

    def is_goal(self, state: tuple[int, int]) -> bool:
        return state == self.goal

    def neighbors(
        self, state: tuple[int, int]
    ) -> list[tuple[int, int]]:
        x, y = state

        candidates = [
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ]

        return [
            pos for pos in candidates
            if self._in_bounds(pos) and pos not in self.walls
        ]

    def step_cost(
        self,
        state: tuple[int, int],
        neighbor: tuple[int, int],
    ) -> float:
        if neighbor not in self.neighbors(state):
            raise ValueError("Invalid transition")

        return 1.0

    def heuristic(self, state: tuple[int, int]) -> float:
        x, y = state
        gx, gy = self.goal

        return abs(x - gx) + abs(y - gy)