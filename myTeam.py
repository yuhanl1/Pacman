# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import distanceCalculator
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'EatAgent', second = 'ProtectAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class SmartAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.heuristicInfo = {}
    #Q-Learning
    self.GAMA = 0.8
    self.scoreReward = 20
    self.foodDistanceReward = 30
    self.capsuleDistanceReward = 50
    self.foodNumReward = 100
    self.capsuleNumReward = 1000
    self.capsuleDefendAward = 1
    self.dangerEnemyNumReward = 999999
    self.dangerEnemyDistanceReward = 10
    self.enemyDistanceReward = 10000

    self.food = self.getFood(gameState)
    self.foodNum = self.getFood(gameState).asList().__len__()

    self.myTeamIndices = self.getTeam(gameState)
    self.enemyIndices = self.getOpponents(gameState)

    self.start = gameState.getAgentPosition(self.index)
    self.returnCount = 0

    self.gridWidth = gameState.data.layout.width
    self.gridHeight = gameState.data.layout.height
    self.wallList = gameState.getWalls().asList()
    self.halfX = self.gridWidth/2 - 2 if self.red else self.gridWidth/2 + 3
    self.middleAreas = [(self.halfX,y) for y in range(self.gridHeight) if (self.halfX,y) not in self.wallList]
    #print self.middleAreas
    self.middleSections = [area for area in self.middleAreas if area[1] in range(self.gridHeight / 4, self.gridHeight /4 * 3)]
    #print self.middleSections
    self.middlePosition = random.choice(self.middleSections)
    #print self.middlePosition

    #print self.middlePosition
    self.legalPositions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
    #print self.legalPositions
    self.middleAvailablePoints = [(self.halfX, y) for y in range(self.gridHeight)
                         if (self.halfX, y) in self.legalPositions]

    self.goToMiddleActionList = self.aStarSearch(gameState, self.heuristic)
    self.goToMiddleActionCount = 0

    self.attack = False
    self.goBack = False


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    currentPosition = gameState.getAgentPosition(self.index)

    myTeam =  self.getTeam(gameState)
    currentNumReturned = 0
    for teamMate in myTeam:
        agentState = gameState.getAgentState(teamMate)
        currentNumReturned += agentState.numReturned

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    #if self.index == 0 or self.index == 1:  #Offensive


    if self.returnCount != currentNumReturned or currentPosition == self.start:
        self.returnCount = currentNumReturned
        self.foodNum = self.getFood(gameState).asList().__len__()
        length = 0
        while length == 0:
            self.goToMiddleActionCount = 0
            self.goToMiddleActionList = self.aStarSearch(gameState, self.heuristic)
            length = self.goToMiddleActionList.__len__()
        self.goToMiddleActionCount += 1
        return self.goToMiddleActionList[self.goToMiddleActionCount - 1]

    if self.goToMiddleActionCount < self.goToMiddleActionList.__len__():
        self.goToMiddleActionCount += 1
        return self.goToMiddleActionList[self.goToMiddleActionCount - 1]
    else:
        if self.index < 2:
            self.goBack = False
            score = self.getScore(gameState)
            currentCarry = gameState.getAgentState(self.index).numCarrying
            if (score < 7 and currentCarry < 6) or (score < 7 and currentCarry < 4):
                self.goBack = False
            else:
                self.goBack = True
            return self.QLearning(gameState, 6)[1]
        else:
            self.attack = False
            dangerEnemies = [enemy for enemy in self.enemyIndices if gameState.getAgentState(enemy).isPacman]
            scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemyIndices]
            #if min(scaredTimes) >= 8 or dangerEnemies.__len__() == 0:
            if  min(scaredTimes) >= 10:
                self.attack = True
            else:
                self.attack = False
            return self.QLearning(gameState, 6)[1]
        '''
        else:
            bestDist = 9999
            goal = random.choice(self.middleAreas)
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(goal, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        '''



        '''
        if self.index < 2:
            return self.QLearning(gameState,3)[1]
        else:
            length = 0
            while length == 0:
                self.goToMiddleActionCount = 0
                self.goToMiddleActionList = self.aStarSearch(gameState, self.heuristic)
                length = self.goToMiddleActionList.__len__()
            self.goToMiddleActionCount += 1
            return self.goToMiddleActionList[self.goToMiddleActionCount - 1]
        '''

    '''
    else:
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)
    '''



  def evaluate(self, gameState, action):

    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def aStarSearch(self, gameState, heuristic):
      """Search the node that has the lowest combined cost and heuristic first."""
      from game import Directions
      openList = util.PriorityQueue()
      closeList = [] #gameState in closeList
      resultActionList = util.Stack()
      bestCost = 0
      #(gameState,direction,bestCost), bestCost
      openList.push(((gameState, Directions.STOP, bestCost), 0), bestCost + heuristic(gameState))
      while not (openList.isEmpty()):
          tmp = openList.pop()
          if (tmp[0][0] not in closeList) or (tmp[0][2] < bestCost):
              closeList.append(tmp[0][0])
              bestCost = tmp[0][2]
              if self.isGoalState(tmp[0][0]): #output All Actions
                  while tmp[1] is not 0:
                      resultActionList.push(tmp[0][1])
                      tmp = tmp[1]
                  returnActionList = []
                  while not (resultActionList.isEmpty()):
                      returnActionList.append(resultActionList.pop())
                  return returnActionList
              #print gameState.getLegalActions(self.index)
              actions = tmp[0][0].getLegalActions(self.index)
              actions.remove(Directions.STOP)
              #print actions
              #print tmp[0][0].getAgentPosition(self.index)
              for action in actions:
                  successorState = (tmp[0][0].generateSuccessor(self.index,action))
                  #print successorState.getAgentPosition(self.index)
                  if successorState not in closeList:
                      tmpNode = ((successorState, action, 1 + tmp[0][2]), tmp)
                      openList.push(tmpNode, 1 + tmp[0][2] + heuristic(successorState))


  def isGoalState(self,gameState):
      '''
      goalFoodNum = self.foodNum - 1
      currentFoodNum = self.getFood(gameState).asList().__len__()
      if currentFoodNum == goalFoodNum or currentFoodNum == 2:
          self.foodNum = currentFoodNum
          return True
      '''
      return False

  def heuristic(self,gameState):

      return 0


  def QLearning(self,gameState,depth):  #return score and action
      depth -= 1

      reward = self.evaluateReward(gameState)
      if depth == 0 or gameState.isOver():
          return (reward,)

      actions = gameState.getLegalActions(self.index)
      actions.remove(Directions.STOP)
      '''
      previousGameState = self.getPreviousObservation()
      previousDirection = previousGameState.getAgentState(self.index).configuration.direction
      rev = Directions.REVERSE[previousDirection]
      '''
      #print rev

      actionScores = [(action,self.QLearning(gameState.generateSuccessor(self.index, action),depth)) for action in actions]
      newActionScores = []
      '''
      for actScore in actionScores:
          if actScore[0] == rev:
              newActionScores.append((actScore[0],(actScore[1][0]*0.5,actScore[1][1])))
          else:
              newActionScores.append(actScore)
      '''
      sortedActionScores = sorted(actionScores,key=lambda x:x[1][0],reverse=True)
      #print actionScores
      maxActionScore = sortedActionScores[0]

      #print maxActionScore
      maxScore = maxActionScore[1][0]
      #print maxScore

      for sortedActionScore in sortedActionScores:
          if sortedActionScore[1][0] == maxScore:
              newActionScores.append(sortedActionScore)

      returnActionScore = random.choice(newActionScores)

      qValue = reward + self.GAMA * maxScore
      #qValue = self.GAMA * maxScore
      #print (qValue,maxActionScore[0])
      return (qValue,returnActionScore[0])



  def evaluateReward(self,gameState):
      return 1.0

  def getMazeDistanceByDic(self, position, goal):
      if self.heuristicInfo.has_key(str(position) + str(goal)):
          return self.heuristicInfo[str(position) + str(goal)]
      else:
          self.heuristicInfo[str(position) + str(goal)] = self.getMazeDistance(position, goal)
          return self.heuristicInfo[str(position) + str(goal)]


class EatAgent(SmartAgent):
    def evaluateReward(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)


        currentScore = self.getScore(gameState)
        #currentScore as large as possible


        enemyPositions = []
        enemyScareTimes = []
        enemyDistances = []
        for enemy in self.enemyIndices:
            enemyState = gameState.getAgentState(enemy)
            if not enemyState.isPacman:
                enemyPositions.append(gameState.getAgentPosition(enemy))
                enemyScareTimes.append(enemyState.scaredTimer)
        #print enemyPositions
        for enemyPosition in enemyPositions:
            if enemyPosition != None:
                enemyDistances.append(self.getMazeDistanceByDic(currentPosition,enemyPosition))
        minEnemyDistance = 0
        if enemyDistances.__len__() > 0: minEnemyDistance = min(enemyDistances)
        if minEnemyDistance >= 4: minEnemyDistance = 0
        #if minEnemyDistance != 0:print minEnemyDistance
        # minEnemyDistance as large as possible

        minEnemyScaredTime = min(enemyScareTimes) if enemyScareTimes.__len__() else 0
        # if Scared, EnemyDistance as small as possible

        foodList = self.getFood(gameState).asList()
        foodNum = foodList.__len__()
        # foodNum as small as possible

        capsuleList = self.getCapsules(gameState)
        capsuleNum = capsuleList.__len__()
        # capsuleNum as small as possible

        foodDistances = [self.getMazeDistanceByDic(currentPosition,foodPosition) for foodPosition in foodList]
        capsuleDistances = [self.getMazeDistanceByDic(currentPosition, capsulePosition) for capsulePosition in capsuleList]
        minFoodDistance = 0
        if foodDistances.__len__() > 0: minFoodDistance =  min(foodDistances)
        # minFoodDistance as small as possible

        minCapsuleDistance = 0
        if capsuleDistances.__len__() > 0: minCapsuleDistance = min(capsuleDistances)
        # minCapsuleDistance as small as possible

        distancesToMiddle = [self.getMazeDistanceByDic(currentPosition, middlePosition)
                             for middlePosition in self.middleAvailablePoints]
        minDistanceToMiddle = min(distancesToMiddle)
        if self.goBack == True:
            reward =  -2 * minDistanceToMiddle \
                      + self.enemyDistanceReward * 5 * minEnemyDistance
        else:
            if minEnemyScaredTime < 6:
                reward = self.scoreReward * currentScore \
                         - self.enemyDistanceReward * minEnemyDistance \
                         - self.foodDistanceReward * minFoodDistance \
                         - self.foodNumReward * foodNum \
                         - self.capsuleDistanceReward * minCapsuleDistance \
                         - self.capsuleNumReward * capsuleNum
            else:
                reward = self.scoreReward * currentScore \
                         + self.enemyDistanceReward * minEnemyDistance \
                         - self.foodDistanceReward * minFoodDistance \
                         - self.foodNumReward * foodNum \
                         - self.capsuleDistanceReward * minCapsuleDistance \
                         - self.capsuleNumReward * capsuleNum
        return reward

    def isGoalState(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        goalPosition = self.middlePosition
        if currentPosition == goalPosition:
            return True
        return False

    def heuristic(self, gameState):
        goal = self.middlePosition
        position = gameState.getAgentPosition(self.index)
        return self.getMazeDistanceByDic(position,goal)




class ProtectAgent(SmartAgent):
    def evaluateReward(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)

        currentScore = self.getScore(gameState)
        # currentScore as large as possible

        dangerEnemies = []
        dangerEnemyPositions = []
        saveEnemyPositions = []
        enemyScareTimes = []
        agentDistances = gameState.getAgentDistances()
        for enemy in self.enemyIndices:
            enemyState = gameState.getAgentState(enemy)
            enemyScareTimes.append(enemyState.scaredTimer)
            enemyPosition = gameState.getAgentPosition(enemy)

            if enemyState.isPacman:
                dangerEnemies.append(enemy)
                if enemyPosition != None:
                    dangerEnemyPositions.append(enemyPosition)
                else:
                    noiseDistance = agentDistances[enemy]
                    enemyDistanceProbabilities = [(legalPos,gameState.getDistanceProb(
                        util.manhattanDistance(legalPos, currentPosition), noiseDistance))
                                                  for legalPos in self.legalPositions]
                    enemyDistanceProbabilities = sorted(enemyDistanceProbabilities, key=lambda x: x[1], reverse=True)
                    dangerEnemyPositions.append(enemyDistanceProbabilities[0][0])
            else:
                if enemyPosition != None:
                    saveEnemyPositions.append(enemyPosition)
                else:
                    noiseDistance = agentDistances[enemy]
                    enemyDistanceProbabilities = [(legalPos,gameState.getDistanceProb(
                        util.manhattanDistance(legalPos, currentPosition), noiseDistance))
                                                  for legalPos in self.legalPositions]
                    enemyDistanceProbabilities = sorted(enemyDistanceProbabilities, key=lambda x: x[1], reverse=True)
                    saveEnemyPositions.append(enemyDistanceProbabilities[0][0])

        dangerEnemyDistances = [self.getMazeDistanceByDic(currentPosition, enemyPosition) for enemyPosition in
                                dangerEnemyPositions]
        minDangerEnemyDistance = 0
        if dangerEnemyDistances.__len__() > 0: minDangerEnemyDistance = min(dangerEnemyDistances)

        saveEnemyDistances = [self.getMazeDistanceByDic(currentPosition, enemyPosition) for enemyPosition in
                                saveEnemyPositions]
        minSaveEnemyDistance = 0
        if saveEnemyDistances.__len__() > 0: minSaveEnemyDistance = min(saveEnemyDistances)
        # minDangerEnemyDistance as small as possible
        # minSaveEnemyDistance as large as possible

        minEnemyScaredTime = min(enemyScareTimes) if enemyScareTimes.__len__() else 0
        # if Scared, can help other eating food

        foodList = self.getFood(gameState).asList()
        foodNum = foodList.__len__()
        # foodNum as small as possible

        foodDefendList = self.getFoodYouAreDefending(gameState).asList()
        foodDefendNum = foodDefendList.__len__()
        # foodDefendNum as large as possible

        capsuleList = self.getCapsulesYouAreDefending(gameState)
        capsuleNum = capsuleList.__len__()
        # capsuleNum as large as possible

        foodDistances = [self.getMazeDistanceByDic(currentPosition, foodPosition) for foodPosition in foodList]
        foodDefendDistances = [self.getMazeDistanceByDic(currentPosition, foodPosition) for foodPosition in foodDefendList]
        capsuleDistances = [self.getMazeDistanceByDic(currentPosition, capsulePosition) for capsulePosition in
                            capsuleList]
        minFoodDistance = 0
        if foodDistances.__len__() > 0: minFoodDistance = min(foodDistances)
        # minFoodDistance as small as possible

        minFoodDefenceDistance = 0
        if foodDefendDistances.__len__() > 0: minFoodDefenceDistance = min(foodDefendDistances)
        # minFoodDefenceDistance as small as possible


        minCapsuleDistance = 0
        if capsuleDistances.__len__() > 0: minCapsuleDistance = min(capsuleDistances)
        # minCapsuleDistance as small as possible


        distancesToMiddle = [self.getMazeDistanceByDic(currentPosition, middlePosition)
                             for middlePosition in self.middleAreas]
        minDistanceToMiddle = min(distancesToMiddle)

        if self.attack == True:
            reward = self.scoreReward * currentScore \
                     + minSaveEnemyDistance \
                     - self.foodDistanceReward * minFoodDistance \
                     - self.foodNumReward * foodNum
        else:
            reward = -(self.capsuleNumReward * minCapsuleDistance
                       # - self.foodNumReward * foodDefendNum
                       # + self.foodDistanceReward * minFoodDefenceDistance
                       + self.dangerEnemyDistanceReward * minDangerEnemyDistance
                       + self.dangerEnemyNumReward * dangerEnemies.__len__())
                 #-200 * minDistanceToMiddle
        print reward
        return reward

    def isGoalState(self, gameState):
        currentPosition = gameState.getAgentPosition(self.index)
        goalPositions = self.middleAreas
        if currentPosition in goalPositions:
            return True
        return False

    def heuristic(self, gameState):
        goal = self.middleAreas

        position = gameState.getAgentPosition(self.index)
        maxD = 9999
        for xy in goal:
            if self.heuristicInfo.has_key(str(position) + str(xy)):
                tmp = self.heuristicInfo[str(position) + str(xy)]
            else:
                self.heuristicInfo[str(position) + str(xy)] = self.getMazeDistance(position, xy)
                tmp = self.heuristicInfo[str(position) + str(xy)]
            if tmp < maxD:
                maxD = tmp
        # print maxD
        return maxD