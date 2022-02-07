# peg-in-a-hole
In this project we attempt to preform the assignment "peg in a hole" using a UR5, 'Universal Robots', robot. We study the case of a boxed-shaped objecte, that is being inserted into a squered-shape hole, using Impedance Control Law.
Commanding the robot to preform the task requires planning a route for the robot to follow, as it doesn't use a search algorithem. While using Impedance Control Law, in order to achive the end result, we allow applying small diturbances into the route provided - making the end effector, or the box it is holding, colide with a surface - and still get the wanted outcome.
As we publish this project, we should note that there is still work to be done. The velocity of each of the robot's engines does not responde to the classic PID law, thus forcing it to be a control law using a P type controller only (while measuring the speed we can see it does not follow the command). While the system is able to preform well in this given scnario, we predict better results if this issue can be resolved (unfortunatly we where not able to locate the sourse of the issue). 

In order to excecute the code in this repository, make sure you have:

    1. ubuntu 20
    2. Python 3.6
    3. mujoco 200
    4. mujoco-py 2.0.2.13
    5. mojoco activation key

make sure to follow this guide for further instructions :https://medium.com/@chinmayburgul/setting-up-mujoco200-on-linux-16-04-18-04-38e5a3524c85

NOTE: as to-date, you cannot run mujoco 200 on Windows. please make sure you preform all the downloads on Linux or that you have an operating virtual machine.

Lilach Biton & Ido Zohar
