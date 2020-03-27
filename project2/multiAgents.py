# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    currentFood=currentGameState.getFood()
    currentFoodList=currentFood.asList()
    ghostDistance=1
    ghostPos=successorGameState.getGhostPositions()
    for ghost in ghostPos:
      ghostDistance=ghostDistance*manhattanDistance(newPos,ghost)

    foodlist = newFood.asList()
    foodDistance = 0
    for food in foodlist:
      foodDistance = foodDistance+manhattanDistance(food, newPos)

    foodDistanceList=[]
    for food in foodlist:
      foodDistanceList.append(manhattanDistance(newPos,food))
    if newPos not in currentFoodList:
      minFoodDistance=min(foodDistanceList)
    else:
      minFoodDistance = 0

    successorGameState.getScore=(ghostDistance)/(foodDistance*minFoodDistance+1)

    return successorGameState.getScore

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    agentNum=gameState.getNumAgents()

    def maxValue(state,currentDepth):
      # print("call max",currentDepth)
      v=float('-inf')
      nextState=state
      nextDepth = currentDepth
      agentIndex=currentDepth%agentNum
      if state.isWin() or state.isLose():
        v=self.evaluationFunction(state)
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex)==None:
            v = self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):

              successor = state.generateSuccessor(agentIndex, action)

              if (currentDepth + 1) % agentNum == 0:
                nextv, nextstate, nextdepth = maxValue(successor, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
              else:
                nextv, nextstate, nextdepth = minValue(successor, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
      # print(currentDepth,agentIndex,v)
      return v, nextState, nextDepth

    def minValue(state,currentDepth):
      # print("call min", currentDepth)
      v=float('inf')
      nextState=state
      nextDepth = currentDepth
      agentIndex = currentDepth % agentNum
      if state.isWin() or state.isLose():
        v=self.evaluationFunction(state)
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex)==None:
            v = self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):
              successor = state.generateSuccessor(agentIndex, action)
              if (currentDepth+1) % agentNum == 0:
                nextv, nextstate, nextdepth = maxValue(successor, currentDepth + 1)
                if nextv < v:
                  v = nextv
                  nextState = nextstate
                  nextDepth=nextdepth
              else:
                nextv, nextstate, nextdepth = minValue(successor, currentDepth + 1)
                if nextv < v:
                  v = nextv
                  nextState = nextstate
                  nextDepth=nextDepth
      # print(currentDepth, agentIndex, v)
      return v,nextState,nextDepth

    vMax=float('-inf')
    pacmanAction=Directions.EAST
    for action in gameState.getLegalActions(0):
      v,nextstate,nextDepth=minValue(gameState.generateSuccessor(0,action),1)
      if v>vMax:
        vMax=v
        pacmanAction=action
      # print("P:",pacmanAction,vMax)
    # Choose one of the best actions


    return pacmanAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    agentNum = gameState.getNumAgents()
    alpha = float('-inf')
    beta = float('inf')

    def maxValue(state, a, b, currentDepth):
      # print("call max",currentDepth,a, b)
      v = float('-inf')
      nextState = state
      nextDepth = currentDepth
      agentIndex = currentDepth % agentNum
      if state.isWin() or state.isLose():
        v = self.evaluationFunction(state)
        return v,nextState,nextDepth
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          return v,nextState,nextDepth
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex) == None:
            v = self.evaluationFunction(state)
            return v, nextState, nextDepth
          else:

            if (currentDepth + 1) % agentNum == 0:
              for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextv, nextstate, nextdepth = maxValue(successor, a, b, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
                a = max(a, v)
                if a >= b:
                  return v, nextState, nextDepth
              return v, nextState, nextDepth
            else:
              for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextv, nextstate, nextdepth = minValue(successor, a, b, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
                  a = max(a, v)
                  if a >= b:
                    return v, nextState, nextDepth
              return v, nextState, nextDepth

      # print(currentDepth,agentIndex,v)
      # return v, a, b, nextState, nextDepth

    def minValue(state, a, b, currentDepth):
      # print("call min", currentDepth,a,b)
      v = float('inf')
      nextState = state
      nextDepth = currentDepth
      agentIndex = currentDepth % agentNum
      if state.isWin() or state.isLose():
        v = self.evaluationFunction(state)
        return v, nextState, nextDepth
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          return v, nextState, nextDepth
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex) == None:
            v = self.evaluationFunction(state)
            return v, nextState, nextDepth
          else:
            if (currentDepth + 1) % agentNum == 0:
              for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextv, nextstate, nextdepth = maxValue(successor,a,b, currentDepth + 1)
                if nextv < v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
                  b = min(b, v)
                  if a >= b:
                    return v, nextState, nextDepth
              return v, nextState, nextDepth
            else:
              for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextv, nextstate, nextdepth = minValue(successor,a,b, currentDepth + 1)
                if nextv < v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextDepth
                  b = min(b, v)
                  if a >= b:
                    return v, nextState, nextDepth
              return v, nextState, nextDepth
      # print(currentDepth, agentIndex, v)
      # return v, a, b, nextState, nextDepth

    vMax = float('-inf')
    pacmanAction = Directions.EAST
    for action in gameState.getLegalActions(0):
      v, nextstate, nextDepth = minValue(gameState.generateSuccessor(0, action), alpha, beta, 1)
      if v > vMax:
        vMax = v
        pacmanAction = action
      alpha = max(alpha, vMax)
      if alpha >= beta:
        # print(pacmanAction, vMax, alpha, beta)
        return pacmanAction
    #   print(action,vMax,alpha,beta)
    # print(pacmanAction)
    return pacmanAction

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
    # util.raiseNotDefined()
    agentNum=gameState.getNumAgents()

    def maxValue(state,currentDepth):
      # print("call max",currentDepth)
      v=float('-inf')
      nextState=state
      nextDepth = currentDepth
      agentIndex=currentDepth%agentNum
      if state.isWin() or state.isLose():
        v=self.evaluationFunction(state)
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex)==None:
            v = self.evaluationFunction(state)
          else:
            for action in state.getLegalActions(agentIndex):

              successor = state.generateSuccessor(agentIndex, action)

              if (currentDepth + 1) % agentNum == 0:
                nextv, nextstate, nextdepth = maxValue(successor, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
              else:
                nextv, nextstate, nextdepth = expValue(successor, currentDepth + 1)
                if nextv > v:
                  v = nextv
                  nextState = nextstate
                  nextDepth = nextdepth
      # print(currentDepth,agentIndex,v)
      return v, nextState, nextDepth

    def expValue(state,currentDepth):
      # print("call min", currentDepth)
      v=0
      nextState=state
      nextDepth = currentDepth
      agentIndex = currentDepth % agentNum
      if state.isWin() or state.isLose():
        v=self.evaluationFunction(state)
      else:
        if currentDepth == (agentNum * self.depth):
          v = self.evaluationFunction(state)
          # print(agentIndex, currentDepth, v)
        else:
          if state.getLegalActions(agentIndex)==None:
            v = self.evaluationFunction(state)
          else:
            actions=state.getLegalActions(agentIndex)
            successorNum=len(actions)
            p=1.0/successorNum
            for action in actions:
              successor = state.generateSuccessor(agentIndex, action)
              if (currentDepth+1) % agentNum == 0:
                nextv, nextstate, nextdepth = maxValue(successor, currentDepth + 1)
                v+=nextv*p
                nextState = nextstate
                nextDepth=nextdepth
              else:
                nextv, nextstate, nextdepth = expValue(successor, currentDepth + 1)
                v += nextv * p
                nextState = nextstate
                nextDepth = nextdepth
      # print(currentDepth, agentIndex, v)
      return v,nextState,nextDepth

    vMax=float('-inf')
    pacmanAction=Directions.EAST
    for action in gameState.getLegalActions(0):
      v,nextstate,nextDepth=expValue(gameState.generateSuccessor(0,action),1)
      if v>vMax:
        vMax=v
        pacmanAction=action
      # print("P:",pacmanAction,vMax)
    # Choose one of the best actions


    return pacmanAction



def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

  """
  "*** YOUR CODE HERE ***"
  # util.raiseNotDefined()



  # if currentGameState.isWin() or currentGameState.isLose():
  #   score=currentGameState.getScore()
  # else:
  #
  #   pacmanPos = currentGameState.getPacmanPosition()
  #   currentFood = currentGameState.getFood()
  #   currentFoodPos = currentFood.asList()
  #   ghostDistance = 1
  #   ghostPos = currentGameState.getGhostPositions()
  #   ghostDistanceList=[]
  #   capsuleDistance = 0
  #   capsulePos=currentGameState.getCapsules()
  #   capsuleDistanceList = []
  #   for ghost in ghostPos:
  #     ghostDistance = ghostDistance * manhattanDistance(pacmanPos, ghost)
  #     ghostDistanceList.append(ghostDistance)
  #   minGhostDistance=min(ghostDistanceList)
  #
  #   foodDistance = 0
  #   foodDistanceList = []
  #   for food in currentFoodPos:
  #     foodDistance = foodDistance + manhattanDistance(food, pacmanPos)
  #     foodDistanceList.append(manhattanDistance(food,pacmanPos))
  #   foodNum=len(foodDistanceList)
  #
  #   if pacmanPos not in currentFoodPos:
  #     minFoodDistance = min(foodDistanceList)
  #   else:
  #     minFoodDistance = 0
  #
  #
  #   if len(capsulePos)==0:
  #     capsuleDistance=0
  #     minCapsuleDistance=0
  #   else:
  #     capsuleDistance = 0
  #     for capsule in capsulePos:
  #       capsuleDistance = capsuleDistance + manhattanDistance(pacmanPos, capsule)
  #       capsuleDistanceList.append(manhattanDistance(pacmanPos, capsule))
  #     minCapsuleDistance=min(capsuleDistanceList)
  #     capsuleNum=len(capsuleDistanceList)
  #
  #   if minGhostDistance > 3:
  #     ghostDistance = minGhostDistance
  #     foodDistance = foodDistance/foodNum * minFoodDistance
  #     capsuleDistance=capsuleNum*minCapsuleDistance
  #     score=ghostDistance/(foodDistance*minFoodDistance+0.1)/(capsuleDistance+0.1)
  #   else:
  #     score = 0
  #
  #   # if minGhostDistance > 3:
  #   #   ghostDistance = pow(ghostDistance * minGhostDistance, 1.0 / 1.5)
  #   #   foodDistance = pow(foodDistance * minFoodDistance, 1)
  #   #   score = ghostDistance / (foodDistance + 0.1)
  #   #   # score=ghostDistance*minFoodDistance/(foodDistance*minFoodDistance+0.1)
  #   # else:
  #   #   score = (ghostDistance) / (foodDistance * minFoodDistance + 1)
  #
  #
  # return score


  pacmanPos=currentGameState.getPacmanPosition()
  foodDistance = 0
  currentFood = currentGameState.getFood()
  currentFoodList = currentFood.asList()
  foodDistanceList = []

  ghostDistance = 1
  ghostPos = currentGameState.getGhostPositions()
  ghostDistanceList = []
  ghostStates = currentGameState.getGhostStates()
  scaredTime=0
  for ghostState in ghostStates:
    scaredTime=ghostState.scaredTimer

  capsuleDistance = 1
  capsulePos=currentGameState.getCapsules()
  capsuleDistanceList = []

  # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  for ghost in ghostPos:
    # ghostDistance = ghostDistance * manhattanDistance(pacmanPos, ghost)
    ghostDistanceList.append(manhattanDistance(pacmanPos, ghost))
  minGhostDistance=min(ghostDistanceList)
  # ghostDistance=pow(ghostDistance,1.0/2)

  if len(capsulePos)==0:
    capsuleDistance=0
    minCapsuleDistance=0
  else:
    for capsule in capsulePos:
      capsuleDistance = capsuleDistance * manhattanDistance(pacmanPos, capsule)
      capsuleDistanceList.append(manhattanDistance(pacmanPos, capsule))
    minCapsuleDistance=min(capsuleDistanceList)
  capsuleNum=len(capsuleDistanceList)


  for food in currentFoodList:
    foodDistance = foodDistance + manhattanDistance(food, pacmanPos)
    foodDistanceList.append(manhattanDistance(pacmanPos, food))
  foodNum=len(foodDistanceList)
  if len(foodDistanceList) != 0:
    minFoodDistance = min(foodDistanceList)
  else:
    minFoodDistance = 0

  if scaredTime==0:
    if minGhostDistance < 3:
      getScore = -4.9/(minGhostDistance+1)+0.2/(minFoodDistance+1)+1.0/(minCapsuleDistance+1)-0.355*foodNum+1.05*currentGameState.getScore()
    else:
      if minCapsuleDistance < 3:
        getScore = -1.0/(minGhostDistance+1)+0.2/(minFoodDistance+1)+5.0/(minCapsuleDistance+1)-0.355*foodNum+1.05*currentGameState.getScore()
      else:
        getScore = -1.0/(minGhostDistance+1)+0.2/(minFoodDistance+1)+1.0/(minCapsuleDistance+1)-0.355*foodNum+1.05*currentGameState.getScore()
  else:
    getScore = -1.0/(minGhostDistance+1)+0.2/(minFoodDistance+1)+1.0/(minCapsuleDistance+1)-0.355*foodNum+1.05*currentGameState.getScore()

  return getScore



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

