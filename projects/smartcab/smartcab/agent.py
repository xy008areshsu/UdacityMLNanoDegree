import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pickle
import os



class QLearning:
    def __init__(self, epsilon = 0.1, alpha = 0.5, gamma = 0.9):
        """
        :param epsilon: probability of doing random move to deal with local stuck
        :param alpha: learning rate
        :param gamma: discount factor for reward
        """
        self.QTable = {} # key: (state, action), value: the Q value
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.possible_actions = Environment.valid_actions
        self.time = 0
        self.alpha_lower_bound = 0.01
        self.epsilon_lower_bound = 0.01
        self.learning_rate_decay_steps = 100
        self.epsilon_decay_steps = 200
        self.learning_rate_decay_factor = 0.8
        self.epsilon_decay_factor = 0.65

    def policy(self, state):
        """
        :param state: current state
        :return: action: argmax over action of QTable[(state, action)], but also small probability of moving randomly
        """
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            vals = [self.QTable[(state, a)] if (state, a) in self.QTable else 0 for a in self.possible_actions]
            keys = [(state, a) for a in self.possible_actions]
            action =  keys[vals.index(max(vals))][1]

        return action

    def update_QTable(self, state, action, reward, next_state):
        if (state, action) not in self.QTable:
            self.QTable[(state, action)] = 0

        new_q = reward + self.gamma * max([self.QTable[(next_state, next_action)]
                                           if (next_state, next_action) in self.QTable else 0
                                           for next_action in self.possible_actions])
        old_q = self.QTable[(state, action)]
        self.QTable[(state, action)] = (1 - self.alpha) * self.QTable[(state, action)] + self.alpha * new_q
        self.time += 1

        # performance learning rate and probability decay
        if self.time % self.learning_rate_decay_steps == 0:
            if self.alpha > self.alpha_lower_bound:
                self.alpha *= self.learning_rate_decay_factor #learning rate decay

        if self.time % self.epsilon_decay_steps == 0:
            if self.epsilon > self.epsilon_lower_bound:
                self.epsilon *= self.epsilon_decay_factor # random move probability decay


        # if abs(self.QTable[(state, action)] - old_q) < 0.1:
        #     print("converged")
        # else:
        #     print('not converged')


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.QLearning = QLearning(epsilon=0.3, alpha=0.95, gamma=0.9)
        # self.QLearning = pickle.load(open('./QLearning.pkl'))

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def is_action_okay(self, inputs):
        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        return action_okay

    def update(self, t):
        # Gather states
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        action_okay = self.is_action_okay(inputs)
        self.state = (action_okay, self.next_waypoint, deadline < 3)
        # self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.QLearning.policy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Gather next states
        next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        next_inputs = self.env.sense(self)
        next_deadline = self.env.get_deadline(self)
        next_action_okay = self.is_action_okay(next_inputs)
        next_state = (next_action_okay, next_waypoint, next_deadline < 3)
        # next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], next_waypoint)

        # TODO: Learn policy based on state, action, reward
        self.QLearning.update_QTable(self.state, action, reward, next_state)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay or add 'display=False' to speed up simulation
    sim.run(n_trials=500)# press Esc or close pygame window to quit
    # pickle.dump(a.QLearning, open(os.path.join('./', 'QLearning.pkl'), 'wb'))

if __name__ == '__main__':
    run()
