# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    init_state = problem.getStartState()

    action = []  # Actions are listed up to each node
    explored = set()  # Explored states are registered for graph search
    frontier = util.Stack()  # LIFO required for Depth-First Search
    frontier.push((init_state, action))  # Initial state is initiated

    state = init_state
    while not problem.isGoalState(state):
        # No solution found and frontier emptied out
        if frontier.isEmpty(): print("FAILURE"); return []
        # The new state visited
        state, action = frontier.pop()
        # If the state is goal state, success!
        if problem.isGoalState(state): return action
        # State saved as explored
        explored.add(state)

        for child_node in problem.getSuccessors(state):
            child_state, child_action, _ = child_node
            # It is not important to have a state in frontier, only checking the explored set
            if child_state in explored: continue
            frontier.push((child_state, action + [child_action]))
            # Implementation from the book doesn't work if we check problem here
            # because we need to expand to make autograder count :/
            # if problem.isGoalState(child_state):
            #     return action + [child_action]
    return action

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    def isIncluded(child_state, list):
        """ Function to find if a state is included in frontier or not
            {child_state: state to be checked if it is in the given frontier list or not,
             list: the frontier list}
        """
        for node in list:
            state, _ = node
            if state == child_state: return True
        return False

    init_state = problem.getStartState()  # Initial state
    action = []  # Actions are listed up to each node
    explored = set()  # Explored states are registered for graph search
    frontier = util.Queue()  # FIFO required for Breadth-First Search
    frontier.push((init_state, action))  # Initial state is initiated

    state = init_state
    while not problem.isGoalState(state):
        if frontier.isEmpty(): print("FAILURE"); return []
        # The new state visited
        state, action = frontier.pop()
        # If the state is goal state, success!
        if problem.isGoalState(state): return action
        # State saved as explored
        explored.add(state)

        for child_node in problem.getSuccessors(state):
            child_state, child_action, _ = child_node
            # Frontier checking is also important now because if a state was in frontier,
            # it would definitely be a shorter way
            if (child_state in explored) or (isIncluded(child_state, frontier.list)): continue
            frontier.push((child_state, action + [child_action]))
    return action

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    def isIncluded(child_state, list):
        """ Function to find if a state is included in frontier or not
            {child_state: state to be checked if it is in the given frontier list or not,
            list: the frontier list}
        """
        for node in list:
            _, _, node_info = node
            state, _ = node_info
            if state == child_state: return True
        return False

    init_state = problem.getStartState()

    action = []
    explored = set()  # Explored states are registered for graph search
    cost_dict = dict()  # Cost info for each state are registered to cost check
    frontier = util.PriorityQueue()  # Priority for smallest cost for Uniform-Cost Search
    frontier.push((init_state, action), 0)  # Initial state is initiated
    cost_dict[init_state] = 0  # Cost of start state = 0

    state = init_state
    while not problem.isGoalState(state):
        if frontier.isEmpty(): print("FAILURE"); return []
        # The new state visited
        state, action = frontier.pop()
        # If the state is goal state, success!
        if problem.isGoalState(state): return action
        # State saved as explored
        explored.add(state)

        for child_node in problem.getSuccessors(state):
            child_state, child_action, child_cost = child_node
            # Cost up to this point is retrieved from the dictionary
            new_child_cost = cost_dict[state] + child_cost
            new_child_action = action + [child_action]
            # It is important to check both explored and frontier set
            check_condition = (child_state in explored) or (isIncluded(child_state, frontier.heap))
            # Check_condition + Check if there was a previous node with higher cost to replace it
            if (check_condition) and (new_child_cost < cost_dict[child_state]):
                # If there is a lower cost path, the frontier is updated to consider the lowest cost path
                frontier.update((child_state, new_child_action), new_child_cost)
                cost_dict[child_state] = new_child_cost
            # Again, it is important to check both explored and frontier set
            if check_condition: continue
            frontier.push((child_state, new_child_action), new_child_cost)
            cost_dict[child_state] = new_child_cost
    return action

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def isIncluded(child_state, list):
        """ Function to find if a state is included in frontier or not
            {child_state: state to be checked if it is in the given frontier list or not,
            list: the frontier list}
        """
        for node in list:
            _, _, node_info = node
            state, _ = node_info
            if state == child_state: return True
        return False

    init_state = problem.getStartState()

    action = []
    explored = set()  # Explored states are registered for graph search
    cost_dict = dict()  # Cost info for each state are registered to cost check
    frontier = util.PriorityQueue()  # Priority is essential for Uniform-Cost Search

    cost_dict[init_state] = 0  # Cost of start state = 0
    priority = cost_dict[init_state] + heuristic(init_state, problem)
    frontier.push((init_state, action), priority)  # Initial state added to frontier

    state = init_state
    while not problem.isGoalState(state):
        if frontier.isEmpty(): print("FAILURE"); return []
        # The new state visited
        state, action = frontier.pop()
        # If the state is goal state, success!
        if problem.isGoalState(state): return action
        # State saved as explored
        explored.add(state)

        for child_node in problem.getSuccessors(state):
            child_state, child_action, child_cost = child_node
            new_child_cost = cost_dict[state] + child_cost
            new_child_action = action + [child_action]
            priority = new_child_cost + heuristic(child_state, problem)
            # It is important to check both explored and frontier set
            check_condition = (child_state in explored) or (isIncluded(child_state, frontier.heap))
            # Check_condition + check if there was a previous node with higher cost to replace it
            if (check_condition) and (new_child_cost < cost_dict[child_state]):
                # If there is a lower cost path, the frontier is updated to get the lowest cost path
                frontier.update((child_state, new_child_action), priority)
                cost_dict[child_state] = new_child_cost
            # Again, it is important to check both explored and frontier set
            if check_condition: continue
            frontier.push((child_state, new_child_action), priority)
            cost_dict[child_state] = new_child_cost
    return action


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
