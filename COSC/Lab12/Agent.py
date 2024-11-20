import os
import re
import numpy as np
import sklearn
import random
from tensorflow import keras as ks
# other imports if needed


class Agent:
    """This class defines your reinforcement learning agent.
    You may add as many helper methods and class fields as needed."""

    model = 0
    actions = []
    observs = []
    rewards = []

    def __init__(self, load_name):
        """Constructor. Loads existing model if one exists, otherwise creates a new model"""
        if load_name:
            self.model = self.load(load_name)
        else:
            self.model = self.create_model()
            self.actions.append([0,0,0])

    def create_model(self):
        """Creates a new untrained deep Q-learning model consisting of one (or more)
        Keras neural networks and stores it in one (or more) class fields (self.field_name)"""
        # CNN to handle observation/state data
        cnn_input = ks.layers.Input(shape=(96, 96, 3), batch_size=32) 
        
        cnn_convLayer = ks.layers.Conv2D(filters=3, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_input) 
        cnn_convLayer = ks.layers.Conv2D(filters=24, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_convLayer)
        cnn_maxPool = ks.layers.MaxPooling2D(pool_size=(2,2)) (cnn_convLayer)

        cnn_centerLayer = ks.layers.Conv2D(filters=24, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_maxPool)
        cnn_centerLayer = ks.layers.Conv2D(filters=48, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_centerLayer)
        cnn_unpool = ks.layers.Conv2DTranspose(filters=48, kernel_size=(2,2), strides=(2,2))(cnn_centerLayer)

        cnn_concatLayer = ks.layers.Concatenate(axis=3)([cnn_convLayer, cnn_unpool])
        cnn_dconLayer = ks.layers.Dropout(0.2) (cnn_concatLayer)
        cnn_dconLayer = ks.layers.Conv2D(filters=24, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_dconLayer)
        cnn_dconLayer = ks.layers.Conv2D(filters=24, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu") (cnn_dconLayer)
        cnn_flatten = ks.layers.Flatten() (cnn_dconLayer)

        # FNN to handle action
        fnn_input = ks.layers.Input(shape=[3,])
        fnn_dense1 = ks.layers.Dense(90, activation="sigmoid")(fnn_input)
        fnn_dense2 = ks.layers.Dense(30, activation="sigmoid")(fnn_dense1)
        fnn_dense3 = ks.layers.Dense(10, activation="sigmoid")(fnn_dense2)

        concat = ks.layers.Concatenate()([cnn_flatten, fnn_dense3])
        output = ks.layers.Dense(1, activation="linear")(concat)

        model = ks.Model(inputs=input, outputs=output)

        return model

    def load(self, load_name):
        """Loads a previously trained model with identifer load_name
        See https://www.tensorflow.org/guide/keras/save_and_serialize"""
        pass


    def save(self, save_name):
        """Saves the model with identifer save_name
        See https://www.tensorflow.org/guide/keras/save_and_serialize"""
        pass

    
    def start_new_game(self):
        """Called when a new run of the game starts
        You may not need this method, but it is called just in case"""
        pass


    def act(self, observation, reward, deploy=False):
        """Called to inform the agent about the current state of the 
        environment and generate an action in response. 

        This method (and helper methods you add) will perform nearly all of the 
        work for the reinforcement learning, including
           1. Setting an exploration policy to discover how actions and observations relate to rewards
           2. Determining what data & labels are needed for model training
           3. Iteratively re-training the model at some frequency to produce better Q-value estimations 

        Arguments:
            observation: Current game screen as shape (96, 96, 3) array
            reward: Integer value +1000/N  for every track section reached (out of N total sections) 
                    and -0.1 for every time step elapsed
            deploy: False (default) if model should be training (exploring)
                    True if model is competing and should be choosing optimal actions based on its model

        Returns a 3-element list or array specifying the next action taken by the agent
            First element --> steering wheel position between -1 (full left) and 1 (full right)
            Second element --> acceleration pedal position between 0 (no acceleration) and 1 (full acceleration)
            Third element --> brake pedal position between 0 (no brake) and 1 (full brake)
            For example, returning [0, 1, 0] would cause full acceration forward
        """
        if len(self.rewards) <= 100:
            action = [random.uniform(-1.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]
            self.actions.append(action)
            self.observs.append(observation)
            self.rewards.append(reward)
            return action
        self.actions.pop(0)
        self.observs.pop(0)
        self.rewards.pop(0)
        predictions = []
        max_Qval = 0

        for i in range(len(self.actions)):
            Qval = self.model.predict(observation, action)
            if (Qval > max_Qval):
                max_Qval = Qval
            predictions.append(Qval)
            

        for action in self.actions:


        return 