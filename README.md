# Acrobat_trainer_OpenAI
code to train an agent to attempt to solve Acrobot from openAI gym


This training process was done with the agent using q learning to learn
from the environment but the model has a hard time learning when it only
uses epsilon greedy due to the model getting stuck hanging down and only
twitching a littl bit with no further progress. In this model I had attempted
to impliment an Intrinsic Curiosity Model(ICM) along side the q learning model.
The ICM would help with the training by promoting the agent to perform actions for
states that it had not seen before in the environment. This implimentation had
seemed to be sucessfull in preventing the agent from getting stuck and was able
to learn quicker than it had before.

The ICM model was modeled from the following paper.
https://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf

