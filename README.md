# Arduino Movement Patterns Learning with Reinforcement Learning for IOT


## Overview

Hardware(Arduino Uno) of Movement Patterns Learning based on Reinforcement Learning related to Robotics Reinforcement Learning Development

1. I Use Arduino Uno as alternative hardware instead of real hardware for Robotic development project. I participated in the robotics development project and my aim is to develop almost same mechanism to use the robotic project. 
The robotics project is to develop robots leaning Movement Patterns based on Reinforcement Learning. I participated the robotics project, developing Movement Patterns Learning system with alternative Hardware(Arduino Uno)



2. The main algorithm I used is Deep Q-Network(DQN). 
Arduino learned <b>Movement Patterns (Down => Push pump => Up</b>). Afetr Arduino can Push pump action, it get reward just like below

<img src="https://user-images.githubusercontent.com/48679574/85190320-3fc61f80-b2f2-11ea-86a3-c4db8dbe44a9.jpg" width="450px"><img src="https://user-images.githubusercontent.com/48679574/85190307-1c9b7000-b2f2-11ea-9459-0f632e795d81.jpg" width="450px">




3. Eplisode training cycle almost followed DQN mechanism. I improved mainly 「value action frunction」. 
While leaning Movement Patterns, DQN episode training cycle and value action curve is just like below:

<img src="https://user-images.githubusercontent.com/48679574/85190464-b0ba0700-b2f3-11ea-9a03-141dc373add2.jpg" width="450px"><img src="https://user-images.githubusercontent.com/48679574/85190474-c62f3100-b2f3-11ea-85ba-c5439e0a11ff.png" width="450px">


4. Through 50 episode, num of success become more than 16 in each step.

<img src="https://user-images.githubusercontent.com/48679574/85190567-87e64180-b2f4-11ea-86b1-09ac1aeb81c9.png" width="550px">



## Environment

- Mac OS Catalina
- Linux
- Python == 3.7
- tensorflow == 2.2.0
- Keras == 2.2.1 
- CPU



## Summary

About This mechanism and logis are written [my blog](https://trafalbad.hatenadiary.jp/entry/2020/06/20/150128)


