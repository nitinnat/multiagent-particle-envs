import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import tensorflow as tf
import cv2
import random


class Scenario(BaseScenario):
    landmark_union = None
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 20
        #num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
        
        """
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
        """
        # make initial conditions
        self.reset_world(world)
        return world

    def process_mnist_data(self):
        rand_num = random.choice(range(0,len(self.x_train)))
        res = cv2.resize(self.x_train[rand_num], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        #inds = np.nonzero(res)
        

        #result = []
        #result.append(np.round((inds[0]-128)/255.0, 2))
        #result.append(np.round((inds[1]-128)/255.0, 2))

        #result = np.unique(np.array(list(zip(result[0], result[1]))), axis=0)
        #result = np.array(random.sample(list(result), 50))
        import imutils
        import matplotlib.pyplot as plt
        cnts = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        c = (c -56)/112
        #cv2.drawContours(res, [c], -1, (255, 0, 0), 2)
        #cv2.imshow("Image", res)
        #cv2.waitKey(0)
        
        c = np.squeeze(c)
        c = np.float32(c)
        #print(c.dtype)
        
        
        return c

    def _calc_union(self, entities):
        """
        Computes the total union covered by all the entities
        """
        from shapely.geometry import Point
        from shapely.ops import cascaded_union

        circles = []
        for entity in entities:
            x, y = entity.state.p_pos
            radius = entity.size
            circles.append(Point(x,y).buffer(radius))

        union = cascaded_union(circles)
        
        return union

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        result = self.process_mnist_data()

        num_landmarks = 1#len(result)

        ## Add landmarks based on the image generated
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.state.p_pos = np.array(result)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0.35, 0.35, 0.85])
        

        #self.landmark_union = self._calc_union(world.landmarks)

            

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on how much landmark area is covered by agents

        rew = 0
        agent_union = self._calc_union(world.agents)
        # The intersection between the landmark union and the agent union will
        # determine how much area is covered.
        #intersection = agent_union.intersection(self.landmark_union)
        
        # Higher the intersection area, the better
        #rew += intersection.area
        
        
        for l in world.landmarks:
            #dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            # compute distance of agent from contour
            dists = [cv2.pointPolygonTest(l.state.p_pos, tuple(a.state.p_pos), True) for a in world.agents]
            rew += sum(dists)
        
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(np.mean(entity.state.p_pos, axis=0) - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            #print(np.mean(other.state.p_pos, axis=0))
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
