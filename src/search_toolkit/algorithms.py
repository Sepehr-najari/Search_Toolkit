from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Generic, TypeVar

from search_toolkit.problems import SearchProblem

State = TypeVar("State")


@dataclass
class SearchResult(Generic[State]):
    path: list[State] | None
    cost: float | None
    expanded: int
    generated: int

    @property
    def found(self) -> bool:
        return self.path is not None


def reconstruct_path(parent, goal):
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def bfs(problem: SearchProblem[State]) -> SearchResult[State]:
    start = problem.initial_state()
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    generated = 1

    while frontier:
        current = frontier.popleft()
        expanded += 1

        if problem.is_goal(current):
            path = reconstruct_path(parent, current)
            return SearchResult(
                path=path,
                cost=float(len(path) - 1),
                expanded=expanded,
                generated=generated,
            )

        for neighbor in problem.neighbors(current):
            if neighbor in parent:
                continue

            parent[neighbor] = current
            frontier.append(neighbor)
            generated += 1

    return SearchResult(None, None, expanded, generated)


def dfs(problem: SearchProblem[State]) -> SearchResult[State]:
    start = problem.initial_state()
    frontier = [start]
    parent = {start: None}
    expanded = 0
    generated = 1

    while frontier:
        current = frontier.pop()
        expanded += 1

        if problem.is_goal(current):
            path = reconstruct_path(parent, current)
            return SearchResult(
                path=path,
                cost=float(len(path) - 1),
                expanded=expanded,
                generated=generated,
            )

        for neighbor in reversed(problem.neighbors(current)):
            if neighbor in parent:
                continue

            parent[neighbor] = current
            frontier.append(neighbor)
            generated += 1

    return SearchResult(None, None, expanded, generated)


def astar(problem: SearchProblem[State]) -> SearchResult[State]:
    start = problem.initial_state()
    tie_breaker = count()

    frontier = [
        (problem.heuristic(start), next(tie_breaker), 0.0, start)
    ]
    parent = {start: None}
    g_score = {start: 0.0}
    expanded = 0
    generated = 1

    while frontier:
        _, _, current_cost, current = heappop(frontier)

        # Ignore outdated priority-queue entries.
        if current_cost != g_score.get(current):
            continue

        expanded += 1

        if problem.is_goal(current):
            return SearchResult(
                path=reconstruct_path(parent, current),
                cost=current_cost,
                expanded=expanded,
                generated=generated,
            )

        for neighbor in problem.neighbors(current):
            tentative_g = (
                current_cost + problem.step_cost(current, neighbor)
            )

            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            g_score[neighbor] = tentative_g
            parent[neighbor] = current

            f_score = tentative_g + problem.heuristic(neighbor)

            heappush(
                frontier,
                (f_score, next(tie_breaker), tentative_g, neighbor),
            )
            generated += 1

    return SearchResult(None, None, expanded, generated)