# valueIterationAgents.py
# -----------------------
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
import queue

# valueIterationAgents.py
# -----------------------
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


import mdp, util
from queue import PriorityQueue

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getGreedyUpdate(self, state):
        """computes a one step-ahead value update and return it"""
        if self.mdp.isTerminal(state):
            return self.values[state]
        actions = self.mdp.getPossibleActions(state)
        vals = util.Counter()
        for action in actions:
            vals[action] = self.computeQValueFromValues(state, action)
        return max(vals.values())

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            qSet = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                qValue = self.computeQValueFromValues(state, self.computeActionFromValues(state))
                qSet[state] = qValue
            self.values = qSet


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transtionProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        # reward = self.mdp.getReward(state, action, )
        res = 0
        for nextState, prob in transtionProbs:
            res += prob * (self.mdp.getReward(state, action, nextState) + self.discount* self.getValue(nextState))

        return res
    

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        
        bestAction = None
        bestValue = float('-inf')
        for action in actions:
            if self.computeQValueFromValues(state, action) > bestValue:
                bestValue = self.computeQValueFromValues(state, action)
                bestAction = action
        return bestAction
            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        self.queue = util.PriorityQueue()
        self.predecessors = util.Counter()

        for state in self.mdp.getStates():
            #compute predecessors of all states
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                transition = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in transition:
                    if prob > 0 and not self.mdp.isTerminal(nextState):
                        if nextState not in self.predecessors:
                            self.predecessors[nextState] = set()
                        self.predecessors[nextState].add(state)
            
            # setup priority queue for all states based on their highest diff in greedy update
            diff = abs(self.values[state] - self.computeQValueFromValues(state, self.computeActionFromValues(state)))
            self.queue.push(state, -diff)
        
        
        # run priority sweeping value iteration:
        for i in range(self.iterations):
            if self.queue.isEmpty():
                break
            state = self.queue.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getGreedyUpdate(state)
            for p in self.predecessors[state]:
                diff = abs(self.values[p] - self.computeQValueFromValues(p, self.computeActionFromValues(p)))
                if diff > self.theta:
                    self.queue.update(p, -diff)
        



class AsynchronousValueIterationAgent:
    print("Not part of this assignment.")
    pass







