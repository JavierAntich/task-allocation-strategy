#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spyder Editor
"""

# LAST UPDATE: 02/10/2022

############
# LIBRARIES
############
import sys
from datetime import datetime
import random as rnd
import matplotlib.pyplot as plt
import math
from enum import Enum
import numpy as np


###############
# ENUMERATIONS
###############
class MDtypes(Enum):

    # Enumeration to classify all types of methods (MD) that can be used to compute the stimulus of a task
    MD_type1 = 1 # "deadline" and "travelling time" are considered together
    MD_type2 = 2 # "deadline" and "travelling time" are considered separately and, later, they are aggregated


class UFtypes(Enum):

    # Enumeration to classify all types of Utility Functions (UF)
    UF_type1 = 1
    UF_type2 = 2


class AOtypes(Enum):

    # Enumeration to classify all types of Aggregation Operators (AO)
    NONE           = 0
    TNORMA_MIN     = 1
    TNORMA_MAX     = 2
    TNORMA_PRODUCT = 3
    TNORMA_YAGER   = 4
    HARMONIC_MEAN  = 5
    OWA_OPERATOR   = 6


class PItypes(Enum):

    # Enumeration to classify all types of Physical Interference (PI) modeling functions
    PI_LINEAR      = 1
    PI_TRAPEZOIDAL = 2
    PI_GAUSSIAN    = 3
    PI_EXPONENTIAL = 4


class SIMstatus(Enum):

    # Enumeration to classify all different status of simulation
    SS_INPROGRESS       = 1
    SS_SUCCESSFULENDING = 2
    SS_UNREASONABLETIME = 3


##################
# GLOBAL SETTINGS
##################

# If True, the standard output is redirected to a file
__PrintToFile__ = True

# If True, the Verbose Mode is activated
__VerboseMode__ = False

# If True, robots behave as they would in the real experiments carried out by Toni/Alberto
__RealExpMode__ = False

# The simulation is finished when the time is greater than simendRegFarthestDL times the value of the farthest deadline
__SimEndRegFarthestDL__ = 10.00

# Size, in pixels, of the environment
__EnvXSize__ = 640
__EnvYSize__ = 480

# Number of tasks
__NumTasks__ = 20

# Minimum distance between tasks
__MinDstBtwTasks__ = 100

# Minimum and maximum task's utility
__MinTaskU__ = 0.75
__MaxTaskU__ = 1.00

# Minimum and maximum task's execution time
__MinTaskET__ = 4
__MaxTaskET__ = 30

# Minimum and maximum task's deadline: as a percentage of the task's execution time
__MinTaskDL__ = 2.50
__MaxTaskDL__ = 4.00

# Maximum number of robots per task
__MaxTaskA__ = 3

# Number of robots
__NumRobots__ = 10

# Minimum robot's velocity: as a percentage of the maximum robot's velocity
__MinRobotVel__ = 0.80

# Minimum and maximum robot's work capacity
__MinRobotWrkCap__ = 1
__MaxRobotWrkCap__ = 2

# If not None, robots are initially located at the position (__IniRobotPosX__, __IniRobotPosY__)
# If None, robots are initially located in a task, which is randomly chosen
__IniRobotPosX__ = None #50
__IniRobotPosY__ = None #50

# If True, it is assumed that robots instantaneously move from one task to another
__InstantaneousTravellingTime__ = False

# If True, it is assumed that robots can change the destination task when they are moving from one task to another
__ChangeTaskWhileTravelling__ = [False, True]

# The program will be executed once with each type of stimulus-computation method (MD)
__MDtypes__ = [MDtypes.MD_type2]

# The program will be executed once with each type of Utility Function (UF)
__UFtypes__ = [UFtypes.UF_type1, UFtypes.UF_type2]

# The program will be executed once with each random seed value
__RandomSeedValues__ = [6, 1005]

# If True, Physical Interference (PI) will be taken into account when computing the stimuli (St)
__PhysicalInterference__ = [True]

# The program will be executed once with each type of Physical Interference (PI) modeling function
__PItypes__ = [PItypes.PI_LINEAR]

# The program will be executed once with each type of Physical Interference (PI) modeling function using different sets of parameter values
# PItypes.PI_LINEAR      has two parameters: GAMMA y BETA
# PItypes.PI_TRAPEZOIDAL has no parameters
# PItypes.PI_GAUSSIAN    has no parameters
# PItypes.PI_EXPONENTIAL has no parameters
__PIparams__ = [[[0.00, 1.00]]]

# If True, Inertia (IN, i.e. tendency to remain at the same task) will be taken into account when computing the stimuli (St)
__Inertia__ = [True]

# The Inertia (IN) modeling function has one parameter: 0 <= K <= 1; the program will be executed once with each given value of K
__INparams__ = [[0.00], [0.33], [0.66], [1.00]]

# The program will be executed once with each pair of Aggregation Operators (AO)
__AOtypes__ = [[AOtypes.TNORMA_MIN, AOtypes.TNORMA_MAX], [AOtypes.TNORMA_PRODUCT, AOtypes.TNORMA_MAX], [AOtypes.HARMONIC_MEAN, AOtypes.TNORMA_MAX], [AOtypes.OWA_OPERATOR, AOtypes.TNORMA_MAX]]

# The program will be executed once with each type of Aggregation Operator (AO) using different sets of parameter values
# AOtypes.TNORMA_MIN     has no  parameters
# AOtypes.TNORMA_MAX     has no  parameters
# AOtypes.TNORMA_PRODUCT has no  parameters
# AOtypes.TNORMA_YAGER   has one parameter: LAMDA > 0
# AOtypes.HARMONIC_MEAN  has no  parameters
# AOtypes.OWA_OPERATOR   has one (WMAX) or two (WMAX, WMIN) parameters: 0 <= WMAX, WMIN <= 1
__AOparams__    = [[[None]]] # In case __PhysicalInterference__ is False
__AOparams_PI__ = [[[None]], [[None]], [[None]], [[1.00, 0.00], [0.75, 0.15], [0.50, 0.33], [1.00/3, 1.00/3]]] # In case __PhysicalInterference__ is True


#######################
# ROBOT MOVEMENT CLASS
#######################
class RobotMovement:

    # ATTRIBUTES
    # decisiontakenat      -> instant of time in which the movement is decided to be taken
    # initialrobotposition -> position of the robot at the moment the movement is decided to be taken

    # "destination task" means the task to which the robot wants to move
    # arrivaltime        -> time of arrival to the destination task
    # finalrobotposition -> position of the destination task    
    # destinationtask    -> internal identifier of the destination task


    # METHODS: Constructor
    def __init__(self, decisiontakenat, arrivaltime, initialrobotposition, finalrobotposition, destinationtask):
        self.decisiontakenat      = decisiontakenat
        self.arrivaltime          = arrivaltime
        self.initialrobotposition = initialrobotposition
        self.finalrobotposition   = finalrobotposition
        self.destinationtask      = destinationtask


###############
# ROBOT CLASS
###############
class Robot:

    # ATTRIBUTES
    # id      -> internal identifier of the robot
    # vel     -> velocity of the robot expressed in pixels per unit time
    # wrkcap  -> How much work can be done by the robot on one task during one unit of time?
    # wrkdone -> Total work done by the robot on tasks
    # mov     -> list of movements made by the robot


    # METHODS:
    # Constructor
    def __init__(self, id, vel, wrkcap, initialpos, firsttask):
        self.id      = id
        self.vel     = vel
        self.wrkcap  = wrkcap
        self.wrkdone = 0
        self.mov     = []
        if (initialpos is None):
            # The robot is initially on a task
            self.mov.append(RobotMovement(0, 0, firsttask.pos, firsttask.pos, firsttask.id))
        else:
            # The robot is located at initialpos
            dst = initialpos.getDistanceTo(firsttask.pos)
            tt  = self.getTravellingTime(dst)
            self.mov.append(RobotMovement(0, tt, initialpos, firsttask.pos, firsttask.id))


    # The robot moves to the highest utility task
    def changeTask(self, departureTime, arrivalTime, initialPosition, finalPosition, destinationTask):
        lastmov = self.mov[-1]
        lastmov.finalrobotposition = initialPosition
        self.mov.append(RobotMovement(departureTime, arrivalTime, initialPosition, finalPosition, destinationTask))


    # When all tasks have been completed, the robot should be stopped
    def stop(self, curTime):
        lastmov = self.mov[-1]
        lastmov.finalrobotposition = self.getCurrentPosition(curTime)


    # Computes the total distance travelled by the robot
    def getTravelledDistance(self):
        dst = 0
        for onemov in self.mov:
            dst = dst + onemov.initialrobotposition.getDistanceTo(onemov.finalrobotposition)

        return(dst)


    # The robot does some work at the task where is currently located
    def doTaskWork(self, tasks, curTime, UFtype):
        idtask = self.getCurrentTask(curTime)
        if (idtask is not None):
            task = tasks.ltasks[idtask]

            updrET = max(task.rET - self.wrkcap, 0)
            wrkdone = task.rET - updrET
            self.wrkdone = self.wrkdone + wrkdone
            task.rET = updrET

            if (task.isCompleted() and wrkdone > 0):
                self.calcAchievedTaskUtility(task, curTime, UFtype)
                task.ITC = curTime


    # Which task needs to be done most urgently?
    def getBestTaskToMove(self, robots, tasks, curTime, IN, INparamset, PI, PItype, PIparamset, changeTWT, MDtype, UFtype, AOtypeset, AOparamset, realexpMode):
        bestTask  = None
        sbestTask = -1
        tbestTask =  0
        lbestTask = [[bestTask, tbestTask]]

        if ((not self.isTravelling(curTime)) or (self.isTravelling(curTime) and changeTWT)):
            posi = self.getCurrentPosition(curTime)

            if (not realexpMode):
                shuffledltasks = rnd.sample(tasks.ltasks, k=len(tasks.ltasks))
            else:
                shuffledltasks = tasks.ltasks

            for taskj in shuffledltasks:
                if (not taskj.isCompleted()):
                    sTask, tTask = self.getTaskStimulus(posi, taskj, robots.getNumberOfRobotsInTask(taskj.id), curTime, IN, INparamset, PI, PItype, PIparamset, MDtype, UFtype, AOtypeset, AOparamset)
                    if ((sTask is not None) and (not (sTask.is_integer() and sTask == 0))):
                        if (sTask > sbestTask):
                            bestTask  = taskj.id
                            sbestTask = sTask
                            tbestTask = tTask
                            lbestTask = [[bestTask, tbestTask]]
                        elif (math.isclose(sTask, sbestTask)):
                            lbestTask.append([taskj.id, tTask])

        # Is the robot's current task in the list?
        curTask = self.getDestinationTask()
        for taskj in lbestTask:
            if ((taskj[0] is not None) and (taskj[0] == curTask)):
                return(taskj)

        return(lbestTask[0])


    # Computes the stimulus, i.e. the urgency for moving the robot to a given task (taskj) at the current time instant
    def getTaskStimulus(self, posi, taskj, taskjcurA, curTime, IN, INparamset, PI, PItype, PIparamset, MDtype, UFtype, AOtypeset, AOparamset):
        tij = 0
        if ((posi.X != taskj.pos.X) or (posi.Y != taskj.pos.Y)):
            dij = posi.getDistanceTo(taskj.pos)
            tij = self.getTravellingTime(dij)

        if (MDtype is MDtypes.MD_type1):
            st, ut = self.StUtFunction(taskj, tij, curTime, UFtype) # Here, it is assumed that the robot is located at taski
            if (PI):
                st = self.doAggregation(AOtypeset[0], AOparamset, st, self.getStPI(PItype, taskjcurA, taskj.maxA, PIparamset[0], PIparamset[1]))

        else:
            st, ut = self.StUtFunction(taskj, 0, curTime, UFtype) # Here, it is assumed that the robot is located at taskj
            if (not PI):
                st = self.doAggregation(AOtypeset[0], AOparamset, st, self.getStTT(tij, taskj.rET))
    
            else:
                if ((AOtypeset[0] is AOtypes.HARMONIC_MEAN) or (AOtypeset[0] is AOtypes.OWA_OPERATOR)):
                    st = self.doAggregation(AOtypeset[0], AOparamset, st, self.getStTT(tij, taskj.rET), self.getStPI(PItype, taskjcurA, taskj.maxA, PIparamset[0], PIparamset[1]))
                else:
                    st = self.doAggregation(AOtypeset[0], AOparamset, st, self.getStTT(tij, taskj.rET))
                    st = self.doAggregation(AOtypeset[0], AOparamset, st, self.getStPI(PItype, taskjcurA, taskj.maxA, PIparamset[0], PIparamset[1]))

        if (IN):
            st = self.doAggregation(AOtypeset[1], [None], st, self.getStIN(self.getCurrentTask(curTime), taskj.id, INparamset[0]))

        return(st, tij)


    # Computes the part of the Stimulus (St) which is related to "Travelling Time" (TT)
    def getStTT(self, tij, rETj):
        return(self.travellingtimeFS(tij / rETj))


    # Computes the part of the Stimulus (St) which is related to "Physical Interference" (PI)
    def getStPI(self, PItype, PIcurAparam, PImaxAparam, PIGammaparam, PIBetaparam):

        if (PItype is PItypes.PI_LINEAR):
            return(max((((PIGammaparam - PIBetaparam) / PImaxAparam) * PIcurAparam) + PIBetaparam, 0.00))

        elif (PItype is PItypes.PI_TRAPEZOIDAL):
            return(0.00)

        elif (PItype is PItypes.PI_GAUSSIAN):
            return(0.00)

        else: # PItypes.PI_EXPONENTIAL
            return(0.00)


    # Computes the part of the Stimulus (St) which is related to "Inertia" (IN)
    def getStIN(self, curTask, idj, k):
        if ((curTask is not None) and (curTask == idj)):
            return(k);
        else:
            return(0.00);


    # It is assumed that the robot has just completed a given task (taskj).
    # Calculates the utility/reward that has been obtained for completing such a task.
    def calcAchievedTaskUtility(self, taskj, curTime, UFtype):
        st, ut = self.StUtFunction(taskj, 0, curTime, UFtype)
        taskj.achU = ut


    # Computes the part of the Stimulus (St) which is related to "Deadlines" (DL), and the Utility (Ut) function
    def StUtFunction(self, taskj, tij, curTime, UFtype):
        stimulus = None
        utility  = None
        expectedTaskjEnd = curTime + tij + math.ceil(taskj.rET / self.wrkcap)

        if (UFtype is UFtypes.UF_type1):
            if (expectedTaskjEnd <= taskj.dl):
                stimulus = taskj.maxU
                utility  = stimulus
            else:
                stimulus = taskj.maxU * ((0.07 * taskj.dl) / ((expectedTaskjEnd - taskj.dl) + (0.07 * taskj.dl)))
                utility  = stimulus

        elif (UFtype is UFtypes.UF_type2):
            if (expectedTaskjEnd <= taskj.dl):
                stimulus = taskj.maxU * ((1.00 * taskj.dl) / ((taskj.dl - expectedTaskjEnd) + (1.00 * taskj.dl)))
                utility  = taskj.maxU
            else:
                stimulus = taskj.maxU * ((0.07 * taskj.dl) / ((expectedTaskjEnd - taskj.dl) + (0.07 * taskj.dl)))
                utility  = stimulus

        return(stimulus, utility)


    # Implementation of a Fuzzy Set (FS) which takes the travelling time into account
    def travellingtimeFS(self, x, n = 1):
        if (x >= 1):
            return(sys.float_info.epsilon)
        else:
            return(max(1 - math.pow(x, n), sys.float_info.epsilon))


    # Combines the x, y and z data values by applying an Aggregation Operator (AO)
    def doAggregation(self, AOtype, AOparamset, x, y, z = None):

        if ((x is None) or (y is None)):
            return(None)

        param1 = AOparamset[0]

        if (z is None): # Only combines the x and y data

            if (AOtype is AOtypes.TNORMA_MIN):
                return(min(x, y))

            elif (AOtype is AOtypes.TNORMA_MAX):
                return(max(x, y))

            elif (AOtype is AOtypes.TNORMA_PRODUCT):
                return(x * y)

            elif (AOtype is AOtypes.TNORMA_YAGER):
                # param1 = lamda; param1 > 0
                # In case param1 is equal to 1, TNORMA_LUKASIEWICZ
                return(max(0.00, 1 - math.pow(math.pow(1-x, param1) + math.pow(1-y, param1), 1/param1)))

            elif (AOtype is AOtypes.HARMONIC_MEAN):
                if ((x.is_integer() and x == 0) or (y.is_integer() and y == 0)):
                    return(0.00)
                else:
                    return(2 / ((1/x) + (1/y)))

            else:
                # AOtypes.OWA_OPERATOR
                # param1 = wmax; 0 <= param1 <= 1
                if ((x.is_integer() and x == 0) or (y.is_integer() and y == 0)):
                    return(0.00)
                else:
                    return((param1 * max(x, y)) + ((1 - param1) * min(x, y)))

        else: # Combines the x, y and z data
              # It is assumed that the operator does not satisfy the commutative property

            if (AOtype is AOtypes.HARMONIC_MEAN):
                if ((x.is_integer() and x == 0) or (y.is_integer() and y == 0) or (z.is_integer() and z == 0)):
                    return(0.00)
                else:
                    return(3 / ((1/x) + (1/y) + (1/z)))

            else:
                param2 = AOparamset[1]

                # AOtypes.OWA_OPERATOR
                # param1 = wmax; 0 <= param1 <= 1
                # param2 = wmid; 0 <= param2 <= 1
                if ((x.is_integer() and x == 0) or (y.is_integer() and y == 0) or (z.is_integer() and z == 0)):
                    return(0.00)
                else:
                    dmax = max(x, y, z)
                    dmin = min(x, y, z)
                    dlst = [dmin, dmax]

                    if ((x in dlst) and (y in dlst)):
                        dmid = z
                    elif ((x in dlst) and (z in dlst)):
                        dmid = y
                    else:
                        dmid = x

                    return((param1 * dmax) + (param2 * dmid) + ((1 - param1 - param2) * dmin))


    # Is the robot currently moving from one task to another?
    def isTravelling(self, curTime):
        if __InstantaneousTravellingTime__:
            return(False)
        else:
            lastmov = self.mov[-1]
            return(curTime < lastmov.arrivaltime)


    # Returns the internal identifier of the task where either the robot is currently going or the robot is currently located
    def getDestinationTask(self):
        lastmov = self.mov[-1]
        return(lastmov.destinationtask)


    # Returns the internal identifier of the task where the robot is currently located
    def getCurrentTask(self, curTime):
        if (not self.isTravelling(curTime)):
            return(self.getDestinationTask())
        else:
            return(None)


    # Returns the current position of the robot
    def getCurrentPosition(self, curTime):
        lastmov = self.mov[-1]
        if (self.isTravelling(curTime)):

            coveredDistance = (curTime - lastmov.decisiontakenat) * self.vel
            totalDistance   = lastmov.initialrobotposition.getDistanceTo(lastmov.finalrobotposition)
            u = (coveredDistance / totalDistance)

            X = lastmov.initialrobotposition.X + ((lastmov.finalrobotposition.X - lastmov.initialrobotposition.X) * u)
            Y = lastmov.initialrobotposition.Y + ((lastmov.finalrobotposition.Y - lastmov.initialrobotposition.Y) * u)
            return(XYtuple(X, Y))
        else:
            return(lastmov.finalrobotposition)


    # Returns the internal identifier of the task to which the robot is currently moving
    def getForthComingTask(self, curTime):
        if (self.isTravelling(curTime)):
            return(self.getDestinationTask())
        else:
            return(None)


    # Returns the time of arrival to the task to which the robot is currently moving
    def getArrivalTime(self, curTime):
        if (self.isTravelling(curTime)):
            lastmov = self.mov[-1]
            return(lastmov.arrivaltime)
        else:
            return(None) 


    # Computes the time that the robot needs to cover a certain distance
    def getTravellingTime(self, distance):
        return(math.ceil(distance / self.vel))


###############
# ROBOTS CLASS
###############
class Robots:

    # ATTRIBUTES
    # lrobots     -> list of robots
    # IN          -> should Inertia (IN) be taken into account when computing the stimuli (St)?
    # INparamset  -> values for the set of parameters of the Inertia (IN) modeling function
    # PI          -> should Physical Interference be taken into account when computing the stimuli (St)?
    # PItype      -> type of Physical Interference (PI) modeling function to be applied
    # PIparamset  -> values for the set of parameters of the Physical Interference (PI) modeling function
    # changeTWT   -> can robots change the destination task when they are moving from one task to another?
    # MDtype      -> type of stimulus-computation method (MD) to be applied
    # UFtype      -> type of Utility Function (UF) to be applied
    # AOtypeset   -> pair of Aggregator Operators (AO) to be applied
    # AOparamset  -> values for the set of parameters of the Aggregation Operator (AO)
    # realexpMode -> if True, robots behave as they would in the real experiments carried out by Toni/Alberto

    # METHODS
    # Constructor   
    def __init__(self, numRobots, envSize, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY, tasks, IN, INparamset, PI, PItype, PIparamset, changeTWT, MDtype, UFtype, AOtypeset, AOparamset, realexpMode):
        self.IN          = IN
        self.INparamset  = INparamset
        self.PI          = PI
        self.PItype      = PItype
        self.PIparamset  = PIparamset
        self.changeTWT   = changeTWT
        self.MDtype      = MDtype
        self.UFtype      = UFtype
        self.AOtypeset   = AOtypeset
        self.AOparamset  = AOparamset
        self.realexpMode = realexpMode
        self.createRobots(numRobots, envSize, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY, tasks)


    # Creates the robots and assigns them a task and a velocity randomly
    def createRobots(self, numRobots, envSize, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY, tasks):

        self.lrobots = []
        numTasks = len(tasks.ltasks)
        taskass  = rnd.sample(range(numTasks), numRobots) # It generates random values without duplicates
                                                          # This means that, at most, there will be one robot assigned to a given task

        maxvelocity = math.ceil(tasks.getMinDistance2Tasks() / 1.25) # We impose a travelling time of, at least, 2 units of time
                                                                     # The divisor can take a value in the range (1, 2]
        minvelocity = math.ceil(maxvelocity * minRobotVel)
        velass    = rnd.choices(range(minvelocity,    maxvelocity+1),    k = numRobots) # It generates random values with duplicates
        wrkcapass = rnd.choices(range(minRobotWrkCap, maxRobotWrkCap+1), k = numRobots)

        for i in range(numRobots):
            if ((iniRobotPosX is None) or (iniRobotPosY is None)):
                self.lrobots.append(Robot(i, velass[i], wrkcapass[i], None, tasks.ltasks[taskass[i]]))
            else:
                self.lrobots.append(Robot(i, velass[i], wrkcapass[i], XYtuple(iniRobotPosX, iniRobotPosY), tasks.ltasks[taskass[i]]))


    # Prints some data of interest about the robots
    def printRobots(self, withmovs):
        print("**************\nROBOTS\n**************\n")
        for robot in self.lrobots:
            print("ROBOT #" + str(robot.id) + "\n\tconstant velocity: " + str(robot.vel) + "\n\twork capacity: " + str(robot.wrkcap), end='')
            if (withmovs):
                print("\n\ttravelled distance: " + str(format(robot.getTravelledDistance(), "8.2f")))
                print("\twork done on tasks: " + str(robot.wrkdone) + "\n")
                print("\t**************\n\tMOVEMENTS\n\t**************\n")

                if (robot.mov[0].arrivaltime == 0):
                    print("\tInitially located at task #" + "{:2d}".format(robot.mov[0].destinationtask) + "\n")
                else:
                    print("\tInitially located at (" + "{:6.2f}".format(robot.mov[0].initialrobotposition.X) + "," + "{:6.2f}".format(robot.mov[0].initialrobotposition.Y) + ")\n")

                for onemov in robot.mov[1:]:
                    if (onemov.decisiontakenat == onemov.arrivaltime):
                        print("\tinstant #" + "{:3d}".format(onemov.decisiontakenat) + " -> stays in task #" + "{:2d}".format(onemov.destinationtask) + "\n")
                    else:
                        print("\tinstant #" + "{:3d}".format(onemov.decisiontakenat) + " -> moves from (" + "{:6.2f}".format(onemov.initialrobotposition.X) + "," + "{:6.2f}".format(onemov.initialrobotposition.Y) + ") to (" + "{:6.2f}".format(onemov.finalrobotposition.X) + "," + "{:6.2f}".format(onemov.finalrobotposition.Y) + ")-task #" + "{:2d}".format(onemov.destinationtask) + " arriving at #" + "{:3d}".format(onemov.arrivaltime) + "\n")
            else:
                print("\n")

        if (withmovs):
            print("Total WORK DONE ON TASKS: " + str(self.getWorkDone()) + "\n")


    # Each robot does some work at the task where is located
    def doRobotsWork(self, tasks, curTime):
        for robot in self.lrobots:
            robot.doTaskWork(tasks, curTime, self.UFtype)


    # Each robot is moved to the task which provides the highest utility
    def moveRobots(self, tasks, curTime):
        if (not self.realexpMode):
            shuffledlrobots = rnd.sample(self.lrobots, k=len(self.lrobots))
        else:
            ireorder = curTime % len(self.lrobots)
            if (ireorder == 0):
                shuffledlrobots = self.lrobots
            else:
                shuffledlrobots = self.lrobots[ireorder:] + self.lrobots[0:ireorder]

        for robot in shuffledlrobots:
            bestTask = robot.getBestTaskToMove(self, tasks, curTime, self.IN, self.INparamset, self.PI, self.PItype, self.PIparamset, self.changeTWT, self.MDtype, self.UFtype, self.AOtypeset, self.AOparamset, self.realexpMode)
            if (bestTask[0] is not None):
                robot.changeTask(curTime, curTime+bestTask[1], robot.getCurrentPosition(curTime), tasks.ltasks[bestTask[0]].pos, bestTask[0])


    # When all tasks have been completed, robots should be stopped
    def stopRobots(self, curTime):
        for robot in self.lrobots:
            robot.stop(curTime)


    # Returns the average distance travelled by the robots
    def getAchievedTravelledDST(self):
        dst = 0
        for robot in self.lrobots:
            dst = dst + robot.getTravelledDistance();

        return(dst / len(self.lrobots))


    # Returns the total work done by robots
    def getWorkDone(self):
        totalW = 0
        for robot in self.lrobots:
            totalW = totalW + robot.wrkdone

        return(totalW)


    # Returns the number of robots that are going to/located in a task
    def getNumberOfRobotsInTask(self, idTask):
        numRobots = 0
        for robot in self.lrobots:
            if (robot.getDestinationTask() == idTask):
                numRobots = numRobots + 1

        return(numRobots)


    # Draws the list of robots which are currently located at each task
    def drawRobots(self, plot, fs, tasks, curTime):
        lbls = []
        for task in tasks:

            nrobotsLoc = 0
            nrobotsTrv = 0
            lblLoc = ""
            lblTrv = ""

            for robot in self.lrobots:
                curTask = robot.getCurrentTask(curTime)
                comTask = robot.getForthComingTask(curTime)

                if ((curTask is not None) and (curTask == task.id)):
                    if (nrobotsLoc):
                        lblLoc = lblLoc + "-" + "A" + str(robot.id) + "(" + str(robot.wrkcap) + "/" + str(robot.wrkdone) + ")"
                    else:
                        lblLoc = "A" + str(robot.id) + "(" + str(robot.wrkcap) + "/" + str(robot.wrkdone) + ")"
                    nrobotsLoc = nrobotsLoc + 1

                elif ((comTask is not None) and (comTask == task.id)):
                    arrTime = robot.getArrivalTime(curTime)
                    if (nrobotsTrv):
                        lblTrv = lblTrv + "-" + "A" + str(robot.id) + "(" + str(arrTime) + ")"
                    else:
                        lblTrv = "A" + str(robot.id) + "(" + str(arrTime) + ")"
                    nrobotsTrv = nrobotsTrv + 1

            if ((nrobotsLoc + nrobotsTrv) > 0):
                if (not task.isCompleted()):
                    plot.add_artist(plt.Circle((task.pos.X, task.pos.Y), radius=2.5*(nrobotsLoc+nrobotsTrv+1), color='green'))
                    plot.add_artist(plt.Circle((task.pos.X, task.pos.Y), radius=2.5*(nrobotsLoc+1), color='red'))
                else:
                    plot.add_artist(plt.Circle((task.pos.X, task.pos.Y), radius=2.5*(nrobotsLoc+nrobotsTrv+1), color='black'))
                    plot.add_artist(plt.Circle((task.pos.X, task.pos.Y), radius=2.5*(nrobotsLoc+1), color='black'))

            if (not task.isCompleted()):
                taskinfoinbrackets = str(task.dl) + "/" + str(task.rET)
            else:
                taskinfoinbrackets = r"$\bf{" + str(format(task.getAchievedUtility(),".0f")) + "\%}$"

            if (nrobotsLoc and nrobotsTrv):
                lbls.append(r"$\bf{T" + str(task.id) + "}$ [" + taskinfoinbrackets + "]: <" + str(nrobotsLoc) + "> " + lblLoc + " || <" + str(nrobotsTrv) + "> " + lblTrv)
            elif (nrobotsLoc):
                lbls.append(r"$\bf{T" + str(task.id) + "}$ [" + taskinfoinbrackets + "]: <" + str(nrobotsLoc) + "> " + lblLoc)
            elif (nrobotsTrv):
                lbls.append(r"$\bf{T" + str(task.id) + "}$ [" + taskinfoinbrackets + "]: || <" + str(nrobotsTrv) + "> " + lblTrv)
            else:
                lbls.append("T" + str(task.id) + " [" + taskinfoinbrackets + "]")

        for robot in self.lrobots:
            if (robot.isTravelling(curTime)):
                robotPos = robot.getCurrentPosition(curTime)
                plot.add_artist(plt.Circle((robotPos.X, robotPos.Y), radius=2.5, color='green'))
                taskPos = tasks[robot.getForthComingTask(curTime)].pos
                plot.arrow(robotPos.X + ((taskPos.X-robotPos.X)/3), robotPos.Y + ((taskPos.Y-robotPos.Y)/3), (taskPos.X-robotPos.X)*(2/3), (taskPos.Y-robotPos.Y)*(2/3), width=1.0, facecolor='none', edgecolor='green')
                plot.arrow(robotPos.X, robotPos.Y, (taskPos.X-robotPos.X)/3, (taskPos.Y-robotPos.Y)/3, width=1.0, head_width=20.0, head_length=10.0, length_includes_head=False, facecolor='green', edgecolor='green')
                plot.text(robotPos.X, robotPos.Y+1.5*fs, "A" + str(robot.id), color='olive', fontsize=fs, horizontalalignment='center', verticalalignment='center')

        return(lbls)


################
# XYTUPLE CLASS
################
class XYtuple:

    # ATTRIBUTES
    # X -> X coordinate
    # Y -> Y coordinate
    
    # METHODS: Constructor        
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    # Computes the Euclidean distance between (X, Y) and a given point
    def getDistanceTo(self, point):
        return(math.sqrt(((self.X - point.X) * (self.X - point.X)) + ((self.Y - point.Y) * (self.Y - point.Y))))


###############
# TASK CLASS
###############
class Task:

    # ATTRIBUTES
    # id   -> internal identifier of the task
    # pos  -> (X,Y)-position of the task
    # maxU -> maximum utility of the task
    # achU -> achieved utility of the task
    # ET   -> execution time to complete the task
    # rET  -> remaining execution time to complete the task
    # dl   -> deadline of the task
    # The condition dl >= ET should be satisfied
    # ITC  -> instant of time in which the task has been completed
    # maxA -> maximum number of robots that can simultaneously be in a task

    # METHODS: Constructor
    def __init__(self, id, pos, maxU, ET, dl, maxA):
        self.id   = id
        self.pos  = pos
        self.maxU = maxU
        self.achU = 0
        self.ET   = ET
        self.rET  = ET
        self.dl   = dl
        self.ITC  = -1
        self.maxA = maxA


    # Has the task been completed?
    def isCompleted(self):
        return(self.rET <= 0)


    # Returns, as a percentage, the utility/reward that was obtained when completing the task
    def getAchievedUtility(self):
        return((self.achU / self.maxU) * 100) # Returns 0 in case the task has not been completed


    # Returns the amount of time the task was completed before the deadline
    def getAchievedTimeBeforeDL(self):
        return(self.dl - self.ITC)


###############
# TASKS CLASS
###############
class Tasks:

    # ATTRIBUTES
    # esize  -> size of the environment
    # ltasks -> list of tasks

    # METHODS
    # Creates the tasks and assigns them a position, an utility, an execution time and a deadline randomly
    def createTasks(self, numTasks, envSize, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxRobotsPerTask):
        self.esize  = envSize
        self.ltasks = []

        lposXY = self.distributeTasks(numTasks, envSize, minDstBtwTasks, 15)
        for i in range(numTasks):
            ET = rnd.randint(minTaskET, maxTaskET)
            dl = round(ET * rnd.uniform(minTaskDL, maxTaskDL))
            self.ltasks.append(Task(i, lposXY[i], rnd.uniform(minTaskU, maxTaskU), ET, dl, maxRobotsPerTask))


    # Computes a random location for each task; two tasks should be located a minimum of minDstBtwTasks apart
    def distributeTasks(self, numTasks, envSize, minDstBtwTasks, margin):
        lposXY = []
        while (len(lposXY) < numTasks):
            posXYaux = XYtuple(rnd.randint(margin, envSize.X-1-margin), rnd.randint(margin, envSize.Y-1-margin))

            check = True
            for posXY in lposXY:
                if (posXYaux.getDistanceTo(posXY) < minDstBtwTasks):
                    check = False

            if (check):
                lposXY.append(posXYaux)

        return(lposXY)


    # Returns the average utility/reward that was obtained when completing all the tasks
    def getAchievedUtility(self):
        sumachUPCT = 0

        for task in self.ltasks:
            achUPCT = task.getAchievedUtility()
            sumachUPCT = sumachUPCT + achUPCT

        return(sumachUPCT / len(self.ltasks))


    # Returns the average amount of time tasks were completed before deadlines
    def getAchievedTimeBeforeDL(self):
        sumachTBDL = 0

        for task in self.ltasks:
            achTBDL = task.getAchievedTimeBeforeDL()
            sumachTBDL = sumachTBDL + achTBDL

        return(sumachTBDL / len(self.ltasks))


    # Tasks that have not been completed are forced to finish
    def finishTasks(self, lrobots, curTime, UFtype):

        for task in self.ltasks:
            if (not task.isCompleted()):
                task.rET = 0
                lrobots[0].calcAchievedTaskUtility(task, curTime, UFtype)
                task.ITC = curTime


    # Prints some data of interest about the tasks
    def printTasks(self, withachU):
        print("**************\nTASKS\n**************\n")
        for task in self.ltasks:
            print("TASK #" + str(task.id) + "\n\tposition: (" + str(task.pos.X) + ", " + str(task.pos.Y) + ")\n\tET:  " + str(task.ET) + "\n\trET: " + str(task.rET) + "\n\tDL:  " + str(task.ET) + " + " + str(task.dl - task.ET) + "\n\tmaxA: " + str(task.maxA) + "\n\tmaxU: " + str(task.maxU), end='')
            if (withachU):
                print("\n\tachU: " + str(task.achU) + "\n\tachU (%): " + str(format(task.getAchievedUtility(),".2f")) + "\n\tITC: " + str(task.ITC) + "\n")
            else:
                print("\n")


    # Draws the position of each task
    def drawTasks(self, plot, fs):
        cirs = []
        for task in self.ltasks:
            if (task.isCompleted()):
                textcolor  = 'black'
                pointcolor = 'black'
            else:
                textcolor  = 'blue'
                pointcolor = 'red'

            cir = plot.add_artist(plt.Circle((task.pos.X, task.pos.Y), radius=2.5, color=pointcolor))
            cirs.append(cir)
            plot.text(task.pos.X, task.pos.Y+1.5*fs, str(task.id), color=textcolor, fontsize=fs, horizontalalignment='center', verticalalignment='center')

        return(cirs)


    # Have all the tasks been completed?
    def areCompleted(self):
        for task in self.ltasks:
            if (not task.isCompleted()):
                return(False)

        return(True)


    # Returns the position of a given task
    def getTaskPosition(self, idTask):
        if ((idTask >= 0) and (idTask < len(self.ltasks))):
            return(self.ltasks[idTask].pos)
        else:
            return(None)


    # Returns the farthest deadline of the tasks which have not been completed yet
    def getFarthestDL(self):
        farthestDL = -1
        for task in self.ltasks:
            if ((not task.isCompleted()) and (task.dl > farthestDL)):
                farthestDL = task.dl

        return(farthestDL)


    # Returns the minimum distance between two tasks
    def getMinDistance2Tasks(self):
        minDistance = sys.float_info.max
        for task1 in self.ltasks:
            for task2 in self.ltasks:
                if (task1.id != task2.id):
                    Distance = task1.pos.getDistanceTo(task2.pos)
                    if (Distance < minDistance):
                        minDistance = Distance

        return(minDistance)


    # Constructor
    def __init__(self, numTasks, envSize, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxRobotsPerTask):
        self.createTasks(numTasks, envSize, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxRobotsPerTask)


###############
# TIME CLASS
###############
class Time:

    # ATTRIBUTES
    # curTime -> discrete current time

    # METHODS
    # Constructor
    def __init__(self):
        self.curTime = 0


    # Returns the current time instant
    def getCurrentTime(self):
        return(self.curTime)


    # Prints the current time instant
    def printCurrentTime(self):
        print("**************\nINSTANT #" + str(self.curTime) + "\n**************")


    # Increases the current time instant by one
    def nextInstant(self):
        self.curTime = self.curTime + 1


#############################
# OPTIMIZATION METHOD CLASS
#############################

class optMethod:

    # ATTRIBUTES
    # simendRegFarthestDL -> The simulation is finished when the current time is greater than simendRegFarthestDL times the value of the farthest deadline
    # clock               -> discrete current time
    # maxSteps            -> number of optimization steps to be taken
    # curStep             -> current optimization step
    # robots              -> robots
    # tasks               -> tasks
    # envSize             -> size of the environment
    # vbMode              -> is the optimization method executed in verbose mode?
    # MDtype              -> type of stimulus-computation method (MD) to be applied
    # IN                  -> should Inertia (IN) be taken into account when computing the stimuli (St)?
    # INparamset          -> values for the set of parameters of the Inertia (IN) modeling function
    # PI                  -> should Physical Interference be taken into account when computing the stimuli (St)?
    # PItype              -> type of Physical Interference (PI) modeling function to be applied
    # PIparamset          -> values for the set of parameters of the Physical Interference (PI) modeling function
    # changeTWT           -> can robots change the destination task when they are moving from one task to another?
    # rndSeed             -> seed value used to produce random numbers
    # UFtype              -> type of Utility Function (UF) to be applied
    # AOtypeset           -> pair of Aggregation Operators (AO) to be applied
    # AOparamset          -> values for the set of parameters of the Aggregation Operator (AO)

    # METHODS
    # Constructor
    def __init__(self, realexpMode, simendRegFarthestDL, MDtype, IN, INparamset, PI, PItype, PIparamset, changeTWT, randomSeedValue, UFtype, AOtypeset, AOparamset, verboseMode, envXSize, envYSize, numTasks, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxTaskA, numRobots, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY):

        self.simendRegFarthestDL = simendRegFarthestDL
        self.MDtype              = MDtype
        self.IN                  = IN
        self.INparamset          = INparamset
        self.PI                  = PI
        self.PItype              = PItype
        self.PIparamset          = PIparamset
        self.changeTWT           = changeTWT
        rnd.seed(randomSeedValue)
        self.rndSeed             = randomSeedValue
        self.UFtype              = UFtype
        self.AOtypeset           = AOtypeset
        self.AOparamset          = AOparamset        
        self.vbMode              = verboseMode
        self.clock               = Time()
        self.envSize             = XYtuple(envXSize, envYSize)

        self.tasks  = Tasks(numTasks, self.envSize, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxTaskA)
        self.robots = Robots(numRobots, self.envSize, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY, self.tasks, self.IN, self.INparamset, self.PI, self.PItype, self.PIparamset, self.changeTWT, self.MDtype, self.UFtype, self.AOtypeset, self.AOparamset, realexpMode)

        if (self.vbMode):
            self.robots.printRobots(False)            
            self.tasks.printTasks(False)


    # Actions involved in an optimization step
    def optStep(self):

        if (self.vbMode):
            self.clock.printCurrentTime()
            self.drawSnapshot(12, self.clock.getCurrentTime())

        self.robots.doRobotsWork(self.tasks, self.clock.getCurrentTime())

        if (self.tasks.areCompleted()):
            return(SIMstatus.SS_SUCCESSFULENDING)
        elif (self.clock.getCurrentTime() > (self.simendRegFarthestDL*self.tasks.getFarthestDL())):
            return(SIMstatus.SS_UNREASONABLETIME)
        else:
            self.robots.moveRobots(self.tasks, self.clock.getCurrentTime())
            self.clock.nextInstant()
            return(SIMstatus.SS_INPROGRESS)


    # Executes the number of optimization steps indicated by maxSteps
    def execute(self):

        while (True):
            SS = self.optStep()
            if (SS is not SIMstatus.SS_INPROGRESS):
                break

        self.robots.stopRobots(self.clock.getCurrentTime())
        self.tasks.finishTasks(self.robots.lrobots, self.clock.getCurrentTime(), self.UFtype)

        if (self.vbMode):
            self.drawSnapshot(12, self.clock.getCurrentTime()+1)
            self.robots.printRobots(True)
            self.tasks.printTasks(True)

        header = "M: " + self.MDTypeToStr(self.MDtype) + " "

        if (self.IN):
            header = header + "IN: Y "
        else:
            header = header + "IN: N "

        if (self.PI):
            header = header + "PI: Y "
        else:
            header = header + "PI: N "

        if (self.AOtypeset[0] is not None):
            if (self.AOtypeset[1] is not AOtypes.NONE):
                header = header + "At: [" + self.AOTypeToStr(self.AOtypeset[0]) + ", " + self.AOTypeToStr(self.AOtypeset[1]) + "] "
            else:
                header = header + "At: [" + self.AOTypeToStr(self.AOtypeset[0]) + "] "

            header = header + "Ap: "
            if (self.AOparamset[0] is None):
                header = header + str(self.AOparamset[0]) + " "
            else:
                for AOparam in self.AOparamset:
                    header = header + str(format(AOparam, ".2f")).replace(".",",") + " "

        if (self.IN):
            header = header + "INp: "
            for INparam in self.INparamset:
                header = header + str(format(INparam, ".2f")).replace(".",",") + " "

        if (self.PI):
            header = header + "PIt: " + self.PITypeToStr(self.PItype) + " "
            header = header + "PIp: "
            for PIparam in self.PIparamset:
                header = header + str(format(PIparam, ".2f")).replace(".",",") + " "

        if (self.changeTWT):
            header = header + "C: Y "
        else:
            header = header + "C: N "

        header = header + "V: " + self.UFTypeToStr(self.UFtype)
        print("<" + self.SIMStatusToStr(SS) + "> " + header, end=' ')
        print("S: " + "{:5d}".format(self.rndSeed), end=' ')

        T = self.clock.getCurrentTime()
        U = self.tasks.getAchievedUtility()
        print("T: " + "{:3d}".format(T), end=' ')
        print("U: " + str(format(U, "6.2f")).replace(".",","), end=' ')

        TbDL = self.tasks.getAchievedTimeBeforeDL()
        print("TbDL: " + str(format(TbDL, "6.2f")).replace(".",","), end=' ')

        D = self.robots.getAchievedTravelledDST()
        print("D: " + str(format(D, "8.2f")).replace(".",","))

        return(header, SS, T, U, TbDL, D)


    # Draws the current state of the optimization system
    def drawSnapshot(self, fs, curTime):

        fig, ax = plt.subplots(ncols=1)
        ax.set_title("SNAPSHOT #" + str(curTime), fontweight="bold", size=fs*2, pad=fs*2)
        ax.set_xlim([0, self.envSize.X-1])
        ax.set_ylim([0, self.envSize.Y-1])
        ax.set_aspect('equal')

        cirs = self.tasks.drawTasks(ax, fs)
        lbls = self.robots.drawRobots(ax, fs*0.8, self.tasks.ltasks, curTime)
        lgd = plt.legend(cirs, lbls, borderaxespad=0, loc='upper left', bbox_to_anchor=(1.1, 1.0)) 
        plt.setp(lgd.texts, family='Courier New')

        plt.show()
        fig.savefig("M" + self.MDTypeToStr(self.MDtype) + "_snapshot" + str(curTime) + ".png", bbox_inches='tight', dpi=200)


    # String conversion of the Aggregation Operator (AO) enum type
    def AOTypeToStr(self, AOtype):

        if (AOtype is AOtypes.NONE):
            return("")
        elif (AOtype is AOtypes.TNORMA_MIN):
            return("TNORMA_MIN")
        elif (AOtype is AOtypes.TNORMA_MAX):
            return("TNORMA_MAX")
        elif (AOtype is AOtypes.TNORMA_PRODUCT):
            return("TNORMA_PRODUCT")
        elif (AOtype is AOtypes.TNORMA_YAGER):
            return("TNORMA_YAGER")
        elif (AOtype is AOtypes.HARMONIC_MEAN):
            return("HARMONIC_MEAN")
        else:
            return("OWA_OPERATOR")


    # String conversion of the Physical Interference (PI) enum type
    def PITypeToStr(self, PItype):

        if (PItype is PItypes.PI_LINEAR):
            return("PI_LINEAR")
        elif (PItype is PItypes.PI_TRAPEZOIDAL):
            return("PI_TRAPEZOIDAL")
        elif (PItype is PItypes.PI_GAUSSIAN):
            return("PI_GAUSSIAN")
        else:
            return("PI_EXPONENTIAL")


    # String conversion of the Utility Function (UF) enum type
    def UFTypeToStr(self, UFtype):
        return(str(UFtype.value))


    # String conversion of the stimulus-computation method (MD) enum type
    def MDTypeToStr(self, MDtype):
        return(str(MDtype.value))


    # String conversion of the Simulation Status (SS) enum type
    def SIMStatusToStr(self, SS):

        if (SS is SIMstatus.SS_INPROGRESS):
            return("SS_INPROGRESS")
        elif (SS is SIMstatus.SS_SUCCESSFULENDING):
            return("SS_SUCCESSFULENDING")
        else:
            return("SS_UNREASONABLETIME")


################
# MAIN FUNCTION
################
def main(printToFile, verboseMode, realexpMode, simendRegFarthestDL, envXSize, envYSize, numTasks, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxTaskA, numRobots, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX = None, iniRobotPosY = None):

    start_time = datetime.now()
    if (printToFile):
        print("Print to file ACTIVATED!\n\n");
        default_stdout = sys.stdout
        filename       = datetime.now().strftime("%Y%m%d_%H%M%S.log")
        sys.stdout     = open(filename, 'w')

    lT    = np.empty(__RandomSeedValues__[1]+1-__RandomSeedValues__[0], dtype='uint32')
    lU    = np.empty(__RandomSeedValues__[1]+1-__RandomSeedValues__[0])
    lTbDL = np.empty(__RandomSeedValues__[1]+1-__RandomSeedValues__[0])
    lD    = np.empty(__RandomSeedValues__[1]+1-__RandomSeedValues__[0])

    for MDtype in __MDtypes__:

        for IN in __Inertia__:
            if (IN):
                __INparams = __INparams__
            else:
                __INparams = [[None]]

            for INparamset in __INparams:
                for PI in __PhysicalInterference__:

                    if ((MDtype is MDtypes.MD_type1) and (not PI) and (not IN)):
                        __AOtypes  = [[None, None]]
                        AOparams__ = [[[None]]]
                    else:
                        __AOtypes     = __AOtypes__
                        AOparams__    = __AOparams__

                    if (PI):
                        __PItypes  = __PItypes__
                        __PIparams = __PIparams__
                        __AOparams = __AOparams_PI__
                    else:
                        __PItypes  = [None]
                        __PIparams = [[[None, None]]]
                        __AOparams = AOparams__

                    i = 0
                    for AOtypeset in __AOtypes:
                        AOparams = __AOparams[i]
                        i = i + 1

                        for AOparamset in AOparams:
                            j = 0
                            for PItype in __PItypes:
                                PIparams = __PIparams[j]
                                j = j + 1

                                for PIparamset in PIparams:
                                    for changeTWT in __ChangeTaskWhileTravelling__:
                                        for UFtype in __UFtypes__:

                                            cForcedEnd = 0
                                            randomSeedRange = range(__RandomSeedValues__[0], __RandomSeedValues__[1]+1)
                                            for randomSeedValue in randomSeedRange:
                                                M = optMethod(realexpMode, simendRegFarthestDL, MDtype, IN, INparamset, PI, PItype, PIparamset, changeTWT, randomSeedValue, UFtype, AOtypeset, AOparamset, verboseMode, envXSize, envYSize, numTasks, minDstBtwTasks, minTaskU, maxTaskU, minTaskET, maxTaskET, minTaskDL, maxTaskDL, maxTaskA, numRobots, minRobotVel, minRobotWrkCap, maxRobotWrkCap, iniRobotPosX, iniRobotPosY)
                                                H, SS, T, U, TbDL, D = M.execute()
                                                if (SS is SIMstatus.SS_UNREASONABLETIME):
                                                    cForcedEnd = cForcedEnd + 1
                                                lT   [randomSeedValue-__RandomSeedValues__[0]] = T
                                                lU   [randomSeedValue-__RandomSeedValues__[0]] = U
                                                lTbDL[randomSeedValue-__RandomSeedValues__[0]] = TbDL
                                                lD   [randomSeedValue-__RandomSeedValues__[0]] = D

                                            print("#" + H)
                                            print("#forcedEND: " + "{:4d}".format(cForcedEnd))
                                            print("#avgT:      " + str(format(np.mean(lT),    "6.2f")).replace(".",",") + " stdT:    " + str(format(np.std(lT),    "6.2f")).replace(".",",") + " varT:    " + str(format(np.var(lT),    "6.2f")).replace(".",","))
                                            print("#avgU:      " + str(format(np.mean(lU),    "6.2f")).replace(".",",") + " stdU:    " + str(format(np.std(lU),    "6.2f")).replace(".",",") + " varU:    " + str(format(np.var(lU),    "6.2f")).replace(".",","))
                                            print("#avgTbDL:   " + str(format(np.mean(lTbDL), "6.2f")).replace(".",",") + " stdTbDL: " + str(format(np.std(lTbDL), "6.2f")).replace(".",",") + " varTbDL: " + str(format(np.var(lTbDL), "6.2f")).replace(".",","))
                                            print("#avgD:      " + str(format(np.mean(lD),    "8.2f")).replace(".",",") + " stdD:    " + str(format(np.std(lD),    "8.2f")).replace(".",",") + " varD:    " + str(format(np.var(lD),    "8.2f")).replace(".",","))
                                            print("#")
                                            # To extract these average results from the output file, you should execute the following console command:
                                            # findstr # filename.log >> res.log

    if (printToFile):
        sys.stdout.close()
        sys.stdout = default_stdout

    end_time = datetime.now()
    print('\nDuration {}'.format(end_time - start_time))
    print("***************\nSIMULATION END!\n***************");


###############
# MAIN PROGRAM
###############

main(__PrintToFile__,
     __VerboseMode__,
     __RealExpMode__,
     __SimEndRegFarthestDL__,
     __EnvXSize__,
     __EnvYSize__,

     __NumTasks__,
     __MinDstBtwTasks__,
     __MinTaskU__,
     __MaxTaskU__,
     __MinTaskET__,
     __MaxTaskET__,
     __MinTaskDL__,
     __MaxTaskDL__,
     __MaxTaskA__,

     __NumRobots__,
     __MinRobotVel__,
     __MinRobotWrkCap__,
     __MaxRobotWrkCap__,
     __IniRobotPosX__,
     __IniRobotPosY__)
