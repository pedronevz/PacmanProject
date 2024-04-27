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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide. You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Ganhar retorna score altissimo
        if successorGameState.isWin():
            return float('+inf')
        
        score = successorGameState.getScore()

        # Distancia de Manhattan entre a posicao atual do Pacman e cada posicao de comida
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            nearestFoodDistance = min(foodDistances)
            # Mais perto da comida, maior o score 
            score -= nearestFoodDistance    

        # Incentivar o pacman a ir comer
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 100 
        
        if currentGameState.getNumFood() < successorGameState.getNumFood():
            score -= 50 

        # Distancia de Manhattan entre a posicao do Pacman e cada posicao de fantasma
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if ghostDistances:
            nearestGhostDistance = min(ghostDistances) 
            if nearestGhostDistance <= 1:
                # Caso pacman seja capturado, retornar uma pontuacao baixissima
                return float('-inf')
            # Mais longe do fantasma, maior o score
            score += nearestGhostDistance    

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # Funcao recursiva minimax para busca da ação a ser tomada
        # agentIndex = 0 p/ pacman
        action, score = self.minimax(gameState, 0, self.depth)
        # minimax retorna melhor ação calculada
        return action
    
    def minimax(self, state, agentIndex, depth):
        # Verifica término da busca
        if state.isWin() or state.isLose() or depth == 0:
            return None, self.evaluationFunction(state)

        # Pacman + fantasmas existentes
        numAgents = state.getNumAgents()
        # Movimento com todos os agentes
        nextAgent = (agentIndex + 1) % numAgents
        # Se o proximo agente a jogar for o pacman, acabou a rodada
        nextDepth = depth - 1 if nextAgent == 0 else depth
        results = []

        # Sobre ações do agente da vez
        for action in state.getLegalActions(agentIndex):
            # Salva estado pos ação
            next_state = state.generateSuccessor(agentIndex, action)
            _, score = self.minimax(next_state, nextAgent, nextDepth)
            results.append((action, score))                     # interessante dar um print(results) para entender as tuplas
            
        # Se o agente atual é o Pacman (max)
        if agentIndex == 0:
            return max(results, key=lambda x: x[1])
        else:  # Se o agente atual é um fantasma (mini)
            return min(results, key=lambda x: x[1])
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Inicializa os valores de alfa e beta conforme pseudo-código do algoritmo
        alpha = float("-inf")
        beta = float("inf")
        bestAction, _ = self.alphabeta(gameState, 0, self.depth, alpha, beta)
        return bestAction
    
    def alphabeta(self, state, agentIndex, depth, alpha, beta):
        #Aqui, mantivemos o código da questão 2, com a adição de alpha-beta pruning
        #A poda reduz significativamente o número de estados que precisam ser explorados durante a busca.
        #elimina a necessidade de explorar ramos que não afetarão o resultado final, tornando-a mais eficiente. 
    
        if state.isWin() or state.isLose() or depth == 0:
            return None, self.evaluationFunction(state)

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth

        if agentIndex == 0: #PACMAN
            return self.maxValue(state, agentIndex, nextAgent, nextDepth, alpha, beta)
        else: #FANTASMAS
            return self.minValue(state, agentIndex, nextAgent, nextDepth, alpha, beta)

    def maxValue(self, state, agentIndex, nextAgent, depth, alpha, beta):
        #Método usado para calcular o valor máximo possível para o agente Pac-Man. 
        bestValue = float("-inf")
        bestAction = None

        #Aqui, atualiza os valores de alfa e beta conforme necessário e retorna a melhor ação e o melhor valor.
        for action in state.getLegalActions(agentIndex):
            next_state = state.generateSuccessor(agentIndex, action)
            _, value = self.alphabeta(next_state, nextAgent, depth, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            if bestValue > beta:  
                return bestAction, bestValue
            alpha = max(alpha, bestValue)  
        return bestAction, bestValue

    def minValue(self, state, agentIndex, nextAgent, depth, alpha, beta):
        #Método usado para calcular o valor mínimo possível para os fantasmas. 
        bestValue = float("inf")
        bestAction = None

        #Diferente do método anterior, busca o valor mínimo e atualiza os valores de alfa e beta de acordo.
        for action in state.getLegalActions(agentIndex):
            next_state = state.generateSuccessor(agentIndex, action)
            _, value = self.alphabeta(next_state, nextAgent, depth, alpha, beta)
            if value < bestValue:
                bestValue = value
                bestAction = action
            if bestValue < alpha:  
                return bestAction, bestValue
            beta = min(beta, bestValue)  
        return bestAction, bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.expectimax(gameState, 0, self.depth)[0]

    def expectimax(self, state, agentIndex, depth):
        #Ao lidar com agentes que tomam decisões de maneira aleatória, usaremos a média das pontuações.
        if state.isWin() or state.isLose() or depth == 0:
            return None, self.evaluationFunction(state), []

        numAgents = state.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == 0 else depth
        results = [] 

        #Ações possíveis do agente da vez
        for action in state.getLegalActions(agentIndex):
            next_state = state.generateSuccessor(agentIndex, action)  #salva estado pos ação
            _, score, next_results = self.expectimax(next_state, nextAgent, nextDepth)  #calcula para o próximo agente
            results.append((action, score))

        #Se é o Pacman (max), retornamos a ação com a maior pontuação
        if agentIndex == 0:
            best_action, best_score = max(results, key=lambda x: x[1])
            return best_action, best_score, results
        else:
            #Se é um fantasma (chance), usa a média das pontuações
            avg_score = sum(score for _, score in results) / len(results)
            return None, avg_score, results

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return float('inf')  # Maximizar a pontuação quando ganhar
    if currentGameState.isLose():
        return float('-inf')  # Minimizar a pontuação quando perder

    score = currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Comida mais proxima
    if foodList:
        nearestFoodDistance = min(manhattanDistance(pacmanPosition, foodPos) for foodPos in foodList)
        score -= nearestFoodDistance  # Incentivar pacman a ir comer

    # Loc. dos fantasmas
    ghostDistances = [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates]
    for distance, scaredTime in zip(ghostDistances, scaredTimes):
        if scaredTime > 0: 
            score += 100 / (distance + 1) # Pontuacao sobe caso pacman se aproxime de um fantasma assustado
        else:
            if distance < 2:
                score -= 1000 # Pontuacao penalizada caso um fantasma esteja muito perto


    # Pontuacao penalizada com a quantidade de comida restante
    score -= (10 * len(foodList))

    return score

# Abbreviation
better = betterEvaluationFunction
