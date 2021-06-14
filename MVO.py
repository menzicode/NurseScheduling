# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""
import random
import numpy
import time
import math
#import shift_scheduling_sat
import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize
from solution import solution



def normr(Mat):
    """normalize the columns of the matrix
    B= normr(A) normalizes the row
    the dtype of A is float"""
    Mat = Mat.reshape(1, -1)
    # Enforce dtype float
    if Mat.dtype != "float":
        Mat = asarray(Mat, dtype=float)

    # if statement to enforce dtype float
    B = normalize(Mat, norm="l2", axis=1)
    B = numpy.reshape(B, -1)
    return B


def randk(t):
    if (t % 2) == 0:
        s = 0.25
    else:
        s = 0.75
    return s


def RouletteWheelSelection(weights):
    accumulation = numpy.cumsum(weights)
    p = random.random() * accumulation[-1]
    chosen_index = -1
    for index in range(0, len(accumulation)):
        if accumulation[index] > p:
            chosen_index = index
            break

    choice = chosen_index

    return choice

# Finds random index in two schedules with desired difference in value
def FindOpposingIndex(SolA, SolB, dif):
    N = len(SolA)
    index = random.randint(0, N-1)
    if SolA[index] - SolB[index] == dif:
        return index
    else:
        FindOpposingIndex(SolA, SolB, dif)

# Finds random index in two schedules with desired difference in value
def FindOpposingIndexRev(SolA, SolB, dif):
    N = len(SolA)
    index = random.randint(0, N-1)
    if SolB[index] + SolA[index] == dif:
        return index
    else:
        FindOpposingIndexRev(SolA, SolB, dif)

def findRandomIndex(solution, value):
    N = len(solution)
    index = random.randint(0, N-1)
    if solution[index] == value:
        return index
    else:
        return findRandomIndex(solution, value)

def findRandomShiftIndex(solution, shift, value):
    N = len(solution) - 1
    index = random.randint(0, N)
    if index % 28 == shift and solution[index] == value:
        return index
    else:
        return findRandomShiftIndex(solution, shift, value)



def MVO(initial_solutions, objf, lb, ub, Max_time, printer):


    "parameters"
    # dim=30
    # lb=-100
    # ub=100
    dim = len(initial_solutions[0])
    N = len(initial_solutions)


    WEP_Max = 1
    WEP_Min = 0.2
    # Max_time=1000
    # N=50
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    #initializes array of universes
    Universes = numpy.zeros((N, dim))
    for i in range(N):
        Universes[i, :] = numpy.array(initial_solutions[i])

    Sorted_universes = numpy.copy(Universes)

    convergence = numpy.zeros(Max_time)

    #initializes best universe variable
    Best_universe = [0] * dim
    Best_universe_Inflation_rate = float("inf")

    s = solution()

    Time = 1
    ############################################
    print('MVO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    while Time < Max_time + 1:

        for i in range(N):
            print(objf(Universes[i]))
        
        print("newit")

        "Eq. (3.3) in the paper"
        WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / Max_time)

        TDR = 1 - (math.pow(Time, 1 / 6) / math.pow(Max_time, 1 / 6))

        #Initializes array of inflation rates
        Inflation_rates = [0] * len(Universes)

        for i in range(0, N):
            for j in range(dim):
                #Check if universes leave search space and bring them back 
                Universes[i, j] = numpy.clip(Universes[i, j], lb[j], ub[j])

            #Evaluate fitness 
            Inflation_rates[i] = objf(Universes[i, :])

            #If this fitness is lower than the best measured, this is now the best measured
            if Inflation_rates[i] < Best_universe_Inflation_rate:

                Best_universe_Inflation_rate = Inflation_rates[i]
                Best_universe = numpy.array(Universes[i, :])

        
        sorted_Inflation_rates = numpy.sort(Inflation_rates)
        sorted_indexes = numpy.argsort(Inflation_rates)

        #Re-sort universes by fitness
        for newindex in range(0, N):
            Sorted_universes[newindex, :] = numpy.array(
                Universes[sorted_indexes[newindex], :]
            )

        normalized_sorted_Inflation_rates = numpy.copy(normr(sorted_Inflation_rates))

        #Universes are sorted
        Universes[0, :] = numpy.array(Sorted_universes[0, :])

        for i in range(1, N):
            
            Black_hole_index = i

            #Universes swap dimensional properties according to inflation rate
            for j in range(0, dim):
                r1 = random.random()
                
                if r1 < normalized_sorted_Inflation_rates[Black_hole_index]:
                    White_hole_index = RouletteWheelSelection(-sorted_Inflation_rates)

                    if White_hole_index == -1:
                        White_hole_index = 0
                   # White_hole_index = 0

                    #shift number 
                    shift = j % 28
                    shiftArrayBlack = []
                    shiftArrayWhite = []
                    shiftIndex = []
                    for q in range(0, dim):
                        if q % 28 == shift:
                            shiftIndex.append(q)
                            shiftArrayBlack.append(Universes[Black_hole_index, q])
                            shiftArrayWhite.append(Sorted_universes[White_hole_index, q])


                    nurse = 0
                    for k in range(len(shiftIndex)):
                        if shiftIndex[k] == j:
                            nurse = k
                    

                    nursesWeek = []
                    for h in range(dim):
                        if shift <= 13:
                            if 28 * nurse <= h and h < 28 * nurse + 14:
                                nursesWeek.append(Universes[Black_hole_index, h])
                        elif shift > 13: 
                            if 28 * nurse + 14 <= h and h < 28 * (nurse + 1):
                                nursesWeek.append(Universes[Black_hole_index, h])

                    shiftNum = 0
                    for i in range(len(nursesWeek)):
                        shiftNum += nursesWeek[i]
                    
                    dayShifts = 0
                    for i in range(len(nursesWeek)):
                        if i % 2 == 0:
                            dayShifts += nursesWeek[i]

                    # Better universe nurse is working, worse isn't
                    # Check if the nurse you are removing will still have >= 1 day shift that week
                    # Check if the nurse you are giving work will have <= 4 shifts that week
                    if shiftArrayBlack[nurse] < shiftArrayWhite[nurse]:
                        validToReplaceIndex = []
                        for i in range(len(shiftIndex)):
                            if i != nurse and shiftArrayBlack[i] == shiftArrayWhite[nurse]:
                                # If the shift is a night shift, any nurse can be moved off the shift 
                                if shift % 2 != 0:
                                    validToReplaceIndex.append(shiftIndex[i])
                                # If the shift is a day shift, check that a nurse will still have >= 1 day shift that week
                                # before moving off
                                else:
                                    weekShifts = []
                                    for h in range(dim):
                                        if shift <= 13:
                                            if 28 * i <= h and h < 28 * i + 14:
                                                weekShifts.append(Universes[Black_hole_index, h])
                                        elif shift > 13:
                                            if 28 * i + 14 <= h and h < 28 * (i + 1):
                                                weekShifts.append(Universes[Black_hole_index, h])
                                    # Count number of day shifts nurse has that week 
                                    dayshifts = 0
                                    for m in range(len(weekShifts)):
                                        if m % 2 == 0:
                                            dayshifts += weekShifts[m]
                                    # If they have more than one day shift, you can move them off it
                                    if dayshifts > 1:
                                        validToReplaceIndex.append(shiftIndex[i])

                        random.shuffle(validToReplaceIndex)
                        replaced = validToReplaceIndex[0]
                        replacedNurse = 0
                        for i in range(len(shiftIndex)):
                            if shiftIndex[i] == replaced:
                                replacedNurse = i
                        Universes[Black_hole_index, replaced] = shiftArrayBlack[nurse]
                        Universes[Black_hole_index, j] = shiftArrayWhite[nurse]

                        # Check if adding this shift for a nurse brings weeks total > 4
                        # If it does, give the nurse who was taken off this shift a shift from this nurse 
                        if shiftNum == 4:
                            replacedWeek = []
                            for h in range(dim):
                                if shift <= 13:
                                    if 28 * replacedNurse <= h and h < 28 * replacedNurse + 14:
                                        replacedWeek.append(Universes[Black_hole_index, h])
                                elif shift > 13: 
                                    if 28 * replacedNurse + 14 <= h and h < 28 * (replacedNurse + 1):
                                        replacedWeek.append(Universes[Black_hole_index, h])                            
                            validShiftToSwitch = []
                            for i in range(len(nursesWeek)):
                                if nursesWeek[i] == shiftArrayWhite[nurse] and replacedWeek[i] != shiftArrayWhite[nurse]:
                                    validShiftToSwitch.append(i)
                            random.shuffle(validShiftToSwitch)
                            if shift <= 13:
                                indexA = 28 * nurse + validShiftToSwitch[0]
                                indexB = 28 * replacedNurse + validShiftToSwitch[0]
                            elif shift > 13:
                                indexA = 28 * nurse + validShiftToSwitch[0] + 14
                                indexB = 28 * replacedNurse + validShiftToSwitch[0] + 14
                            Universes[Black_hole_index, indexA] = shiftArrayBlack[nurse]
                            Universes[Black_hole_index, indexB] = shiftArrayWhite[nurse]

                    # Better universe nurse isn't working, worse is 
                    # Check if the nurse you are removing will still have >= 1 day shift that week
                    # Check if the nurse you are giving work will still have <= 4 shifts that week
                    if shiftArrayWhite[nurse] < shiftArrayBlack[nurse]:
                        lessThanFourIndex = []
                        anyLessThanFour = False
                        for i in range(len(shiftIndex)):
                            if i != nurse and shiftArrayBlack[i] == shiftArrayWhite[nurse]:
                                weekShifts = []
                                for h in range(dim):
                                    if shift <= 13:
                                        if 28 * i <= h and h < 28 * i + 14:
                                            weekShifts.append(Universes[Black_hole_index, h])
                                    elif shift > 13:
                                        if 28 * i + 14 <= h and h < 28 * (i + 1):
                                            weekShifts.append(Universes[Black_hole_index, h])
                                # Count number of shifts nurse has that week 
                                shifts = 0
                                for m in range(len(weekShifts)):
                                    shifts += weekShifts[m]
                                if shifts < 4:
                                    anyLessThanFour = True
                                    lessThanFourIndex.append(shiftIndex[i])
                        if anyLessThanFour:
                            random.shuffle(lessThanFourIndex)
                            given = lessThanFourIndex[0]
                            Universes[Black_hole_index, j] = shiftArrayWhite[nurse]
                            Universes[Black_hole_index, given] = shiftArrayBlack[nurse]
                        
                r2 = random.random()
                
                #Worm holes appeear to randomly distribute dimensions from best universe
                # if r2 < WEP:
                #     BestVal = Best_universe[j]
                #     UniVal = Universes[i, j]
                #     Universes[i, j] = Best_universe[j] # random.uniform(0,1)+lb);



        convergence[Time - 1] = Best_universe_Inflation_rate
        if Time % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(Time)
                    + " the best fitness is "
                    + str(Best_universe_Inflation_rate)
                ]
            )
            # for universe in Universes:
                # print(Best_universe - universe)
                # sum = 0
                # for i in range(len(universe)):
                #     sum += universe[i]
                # print(sum)
            # print("Best")
            # printer(Best_universe)
            # print("Others")
            # for uni in Universes:
            #     printer(uni)

        Time = Time + 1
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "MVO"
    s.objfname = objf.__name__

    return s
