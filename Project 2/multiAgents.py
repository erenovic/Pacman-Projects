# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Eating a food gives 10 pts
        # Each motion loses 1 pts
        # Eating a ghost gives 200 pts
        # Losing gives 400 pts
        # Winning gives 500 pts
        # Getting old position to compare if the action is stop or not
        # (I have not used "Stop" to prevent hard coding)
        # Finding ghostStates to measure the minimum distance for evaluation of score
        ghostStates = [state.getPosition() for state in newGhostStates]
        # Food position list to find the minimum distance for evaluation of score
        newFoodPosList = newFood.asList()
        # Old food position list to increase the score
        # if action decreases the amount of food on the map
        oldFoodPosList = currentGameState.getFood().asList()
        oldNumFoods = len(oldFoodPosList)
        newNumFoods = len(newFoodPosList)
        foodEaten = oldNumFoods - newNumFoods
        # Distance from foods & ghosts for evaluation of score
        distFromFoods = [util.manhattanDistance(newPos, foodPos) for foodPos in newFoodPosList]
        distFromGhosts = [util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostStates]
        # Minimum distance from ghosts used as a part of function
        min_distFromGhosts = min(distFromGhosts)
        # Minimum distance from foods used as a part of function
        # but if there are no foods left, then to prevent error,
        # minimum distance is 0.
        if len(distFromFoods)!=0:
            min_distFromFoods = min(distFromFoods)
        else:
            min_distFromFoods = 0
        # If there are no ghosts, the function does
        # not include distance from ghosts
        if len(distFromGhosts) != 0:
            # If ghosts are scared, Pacman chases the ghosts in enough distance
            if min(newScaredTimes) != 0 and min_distFromGhosts <= min(newScaredTimes):
                score = 200/min_distFromGhosts + 1/min_distFromFoods + 10*foodEaten
            else:
                # If there is a certainty to die in a state, it has the worst score
                if min_distFromGhosts in [0, 1]:
                    return -400
                else:
                    # If there is no risk of death and the action finishes all
                    # foods, then state has a high score
                    if min_distFromFoods == 0:
                        return 500
                    else:
                        # If Pacman is in a safe distance, then it freely consumes food,
                        # if not, it acts more careful. To satisfy continuity I have added 4.
                        if min_distFromGhosts >= 2:
                            score = 10/min_distFromFoods + 10*foodEaten + 4
                        else:
                            score = min_distFromGhosts + 10/min_distFromFoods + 10*foodEaten
        else:
            score = 1/min_distFromFoods + foodEaten
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def value(state, agent, depth):
            if depth == self.depth:
                # Then depth reached
                return self.evaluationFunction(state)
            if state.isWin() or state.isLose():
                # Then terminal state reached, we get the exact score
                return state.getScore()
            if agent == 0:
                # Then it is Pacman and the next agent is MAX
                return maxvalue(state, agent, depth)
            if agent != 0:
                # Then it is Ghost and the next agent is MIN
                # If this is the last agent in the game, depth is increased
                if agent == numAgents-1: depth = depth + 1
                return minvalue(state, agent, depth)

        def minvalue(state, agent, depth):
            v = 9999
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1)%numAgents
                v = min(v, value(successorState, successorAgent, depth))
            return v

        def maxvalue(state, agent, depth):
            v = -9999
            possibleActions = state.getLegalActions(agent)
            for idx, action in enumerate(possibleActions):
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1)%numAgents
                successorValue = value(successorState, successorAgent, depth)
                # If it is the best option for Pacman it saves both the action
                # and the highest value
                if successorValue >= v:
                    v = successorValue
                    bestAction = action
            # If we checked all actions and the agent is Pacman,
            # we return the action, else we return the value. That way if
            # there are other Pacmans working synergeticly,
            # the MinimaxAgent class would still work.
            if (depth == 0) & (agent==0):
                return bestAction
            return v

        return value(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def value(state, agent, depth, alpha, beta):
            if depth == self.depth:
                # Then depth reached
                return self.evaluationFunction(state)
            if state.isWin() or state.isLose():
                # Then terminal state reached, we get the exact score
                return state.getScore()
            if agent == 0:
                # Then it is Pacman and the next agent is MAX
                return maxvalue(state, agent, depth, alpha, beta)
            if agent != 0:
                # Then it is Ghost and the next agent is MIN
                # If this is the last agent in the game, depth is increased
                if agent == numAgents-1: depth = depth+1
                return minvalue(state, agent, depth, alpha, beta)

        def minvalue(state, agent, depth, alpha, beta):
            v = 9999
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1) % numAgents
                v = min(v, value(successorState, successorAgent, depth, alpha, beta))
                # Pruning for v<beta (meaning -> Pacman would never choose v<alpha)
                if v < alpha: return v
                beta = min(beta, v)
            return v

        def maxvalue(state, agent, depth, alpha, beta):
            v = -9999
            possibleActions = state.getLegalActions(agent)
            for idx, action in enumerate(possibleActions):
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1) % numAgents
                successorValue = value(successorState, successorAgent, depth, alpha, beta)
                # If it is the best option for Pacman it saves both the action
                # and the highest value
                if successorValue >= v:
                    v = successorValue
                    bestAction = action
                # Pruning for v>beta (meaning -> Ghosts would not allow v>beta)
                if v > beta: return v
                alpha = max(alpha, v)
            # If we checked all actions and the agent is Pacman,
            # we return the action, else we return the value. That way if
            # there are other Pacmans working synergeticly,
            # the MinimaxAgent class would still work.
            if (depth == 0) & (agent == 0):
                return bestAction
            return v

        return value(state=gameState, agent=0, depth=0, alpha=-9999, beta=9999)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def value(state, agent, depth):
            if depth == self.depth:
                # Then depth reached
                return self.evaluationFunction(state)
            if state.isWin() or state.isLose():
                # Then terminal state reached, we get the exact score
                return state.getScore()
            if agent == 0:
                # Then it is Pacman and the next agent is MAX
                return maxvalue(state, agent, depth)
            if agent != 0:
                # Then it is Ghost and the next agent is MIN
                # If this is the last agent in the game, depth is increased
                if agent == numAgents - 1: depth = depth + 1
                return expvalue(state, agent, depth)

        def expvalue(state, agent, depth):
            v = 9999
            score = 0
            possibleActions = state.getLegalActions(agent)
            n_possibleActions = len(possibleActions)
            for action in possibleActions:
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1) % numAgents
                successorValue = value(successorState, successorAgent, depth)
                score += (1/n_possibleActions) * successorValue
            return score

        def maxvalue(state, agent, depth):
            v = -9999
            possibleActions = state.getLegalActions(agent)
            for idx, action in enumerate(possibleActions):
                successorState = state.generateSuccessor(agent, action)
                # Increasing the agent to pass to the next agent using value().
                successorAgent = (agent + 1) % numAgents
                successorValue = value(successorState, successorAgent, depth)
                # If it is the best option for Pacman it saves both the action
                # and the highest value
                if successorValue >= v:
                    v = successorValue
                    bestAction = action
            # If we checked all actions and the agent is Pacman,
            # we return the action, else we return the value. That way if
            # there are other Pacmans working synergeticly,
            # the MinimaxAgent class would still work.
            if (depth == 0) & (agent == 0):
                return bestAction
            return v
        result = value(state=gameState, agent=0, depth=0)
        return result

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: If Ghosts are scared and Pacman is close enough (wrt. scaredTime),
    he tries to get close and finally eat them. If Ghosts are not scared, Pacman tries
    to collect as many food as possible while keeping the distance close to the
    closest food and being just a bit afraid of ghosts. If the distance with ghosts
    gets too low, he envisions a negative score and runs away without thinking anything else.
    """
    "*** YOUR CODE HERE ***"
    # Eating a food gives 10 pts
    # Each motion loses 1 pts
    # Eating a ghost gives 200 pts
    # Losing gives 400 pts
    # Winning gives 500 pts
    Pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostStartStates = [ghostS.start.pos for ghostS in ghostStates]
    # Finding ghostStates to measure the minimum distance for evaluation of score
    ghostPosList = [state.getPosition() for state in ghostStates]
    # Food position list to find the minimum distance for evaluation of score
    foodPosList = currentGameState.getFood().asList()
    numFoods = len(foodPosList)

    capsulePosList = currentGameState.getCapsules()
    # Distance from foods & ghosts for evaluation of score
    distFromFoods = [util.manhattanDistance(Pos, foodPos) for foodPos in foodPosList]
    distFromGhosts = [util.manhattanDistance(Pos, ghostPos) for ghostPos in ghostPosList]
    distFromCapsules = [util.manhattanDistance(Pos, capsulePos) for capsulePos in capsulePosList]

    numCapsules = len(distFromCapsules)
    # Minimum distance from ghosts used as a part of function
    min_distFromGhosts = min(distFromGhosts)

    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Minimum distance from foods used as a part of function
    min_distFromFoods = min(distFromFoods)

    # If there are no ghosts, the function does
    # not include distance from ghosts
    if len(distFromGhosts) != 0:
        # If ghosts are scared, Pacman follows them
        # (This part is a bit problematic but works often, Pacman sometimes doesn't eat ghosts when nearby)
        if min(scaredTimes) != 0 and min_distFromGhosts <= min(scaredTimes):
            score = 21 / min_distFromGhosts + 1 / min_distFromFoods + 10 / numFoods + 100 / (numCapsules + 1)
        else:
            # If there is a risk to die in a state, it has the worst score
            if (min_distFromGhosts <= 3) or currentGameState.isLose():
                return -500
            else:
                # If Pacman is in a safe distance, then it freely consumes food
                score = 0.5 / min_distFromFoods + 200 / numFoods + 100 / (numCapsules + 1) + 0.005*min_distFromGhosts
    else:
        score = 1 / min_distFromFoods + 1 / numFoods

    return score
# Abbreviation
better = betterEvaluationFunction
