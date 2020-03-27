# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    currentposition = problem.getStartState()
    allvisitedpositions = []
    allvisitedpositions.append(currentposition)
    actions = []
    allexpandedpositions=[]
    path = allvisitedpositions
    while problem.isGoalState(currentposition) == 0:
        successors = problem.getSuccessors(currentposition)
        if successors == None:
            path.pop()
            actions.pop()
        else:
            allexpandedpositions.append(currentposition)
            subposition =[]
            subaction = []
            for position, action, cost in successors:
                subposition.append(position)
                subaction.append(action)
                if position in allexpandedpositions:
                    subposition.pop()
                    subaction.pop()
                else:
                    continue
            if len(subaction) == 0:
                path.pop()
                actions.pop()
            else:
                allvisitedpositions = list(set(allvisitedpositions + subposition))
                allexpandedpositions.append(subposition[-1])
                path.append(subposition[-1])
                actions.append(subaction[-1])
            currentposition = path[-1]
    return actions


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    """
        BFS is an example of UCS. So I just use the solution of UCS here.
    
    """

    start=problem.getStartState()
    allvisitedpositions=[start]
    costdict={start:0}
    actiondict={start:[]}
    currentposition=start
    while problem.isGoalState(currentposition) == 0:
        successors = problem.getSuccessors(currentposition)
        for position, action, cost in successors:
            if position not in allvisitedpositions:
                allvisitedpositions.append(position)
                action_copy = actiondict.copy()
                action_copy[position] = action_copy[currentposition] + [action]
                actiondict = action_copy.copy()
                cost_copy = costdict.copy()
                cost_copy[position]=cost_copy[currentposition]+cost
                costdict=cost_copy.copy()
            else:
                if position in costdict.keys():
                    oldcost = costdict[position]
                    cost_copy1 = costdict.copy()
                    newcost=cost_copy1[currentposition]+cost
                    if oldcost>newcost:
                        action_copy = actiondict.copy()
                        action_copy[position] = action_copy[currentposition] + [action]
                        actiondict = action_copy.copy()
                        cost_copy = costdict.copy()
                        cost_copy[position] = cost_copy[currentposition] + cost
                        costdict = cost_copy.copy()
        costdict.pop(currentposition)
        currentposition=min(costdict,key=lambda x:costdict[x])
    actions=actiondict[currentposition]
    return actions


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    start=problem.getStartState()
    allvisitedpositions=[start]
    costdict={start:0}
    actiondict={start:[]}
    currentposition=start
    while problem.isGoalState(currentposition) == 0:
        successors = problem.getSuccessors(currentposition)
        for position, action, cost in successors:
            if position not in allvisitedpositions:
                allvisitedpositions.append(position)
                action_copy = actiondict.copy()
                action_copy[position] = action_copy[currentposition] + [action]
                actiondict = action_copy.copy()
                cost_copy = costdict.copy()
                cost_copy[position]=cost_copy[currentposition]+cost
                costdict=cost_copy.copy()
            else:
                if position in costdict.keys():
                    oldcost = costdict[position]
                    cost_copy1 = costdict.copy()
                    newcost=cost_copy1[currentposition]+cost
                    if oldcost>newcost:
                        action_copy = actiondict.copy()
                        action_copy[position] = action_copy[currentposition] + [action]
                        actiondict = action_copy.copy()
                        cost_copy = costdict.copy()
                        cost_copy[position] = cost_copy[currentposition] + cost
                        costdict = cost_copy.copy()
        costdict.pop(currentposition)
        currentposition=min(costdict,key=lambda x:costdict[x])
    actions=actiondict[currentposition]
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    start=problem.getStartState()
    allvisitedpositions=[start]
    costdict={start:0}
    totalcostdict ={start:heuristic(start,problem)}
    actiondict={start:[]}
    currentposition=start
    while problem.isGoalState(currentposition) == 0:
        successors = problem.getSuccessors(currentposition)
        for position, action, cost in successors:
            if position not in allvisitedpositions:
                allvisitedpositions.append(position)
                action_copy = actiondict.copy()
                action_copy[position] = action_copy[currentposition] + [action]
                actiondict = action_copy.copy()
                cost_copy = costdict.copy()
                cost_copy[position]=cost_copy[currentposition]+cost
                costdict = cost_copy.copy()
                cost_position=costdict[position]
                totalcostdict[position] = cost_position+heuristic(position,problem)
            else:
                if position in costdict.keys():
                    oldcost = costdict[position]
                    cost_copy1 = costdict.copy()
                    newcost=cost_copy1[currentposition]+cost
                    if oldcost>newcost:
                        action_copy = actiondict.copy()
                        action_copy[position] = action_copy[currentposition] + [action]
                        actiondict = action_copy.copy()
                        cost_copy = costdict.copy()
                        cost_copy[position] = cost_copy[currentposition] + cost
                        costdict = cost_copy.copy()
                        cost_position = costdict[position]
                        totalcostdict[position] = cost_position + heuristic(position, problem)
        costdict.pop(currentposition)
        totalcostdict.pop(currentposition)
        currentposition=min(totalcostdict,key=lambda x:totalcostdict[x])
    actions=actiondict[currentposition]
    return actions



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
