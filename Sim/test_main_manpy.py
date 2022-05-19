from Source import Source
from Queue import Queue
from Machine import Machine
from Exit import Exit
from Failure import Failure
from Repairman import Repairman
from Globals import runSimulation,stepSimulation
from Conveyer import Conveyer
import random
#Machine, Source, Exit, Part, Repairman, Queue, Failure 

def Teste_D():


    #define the objects of the model 
    S1=Source('S1','Source',interarrivalTime={'distributionType':'Fixed','mean':0.5}, entity='Part')
    Q1=Queue('Q1','Queue', capacity=1)
    M1=Machine('M1','Machine', processingTime={'distributionType':'Fixed','mean':0.25})
    E1=Exit('E1','Exit')

    S1.defineRouting(successorList=[Q1])
    Q1.defineRouting(predecessorList=[S1],successorList=[M1])
    M1.defineRouting(predecessorList=[Q1],successorList=[E1])
    E1.defineRouting(predecessorList=[M1])

    runSimulation(objectList=[S1,Q1,M1,E1], maxSimTime=1440.0)

    working_ratio = (M1.totalWorkingTime/1440.0)*100

    print ("the system produced", E1.numOfExits, "parts")
    print ("the total working ratio of the Machine is", working_ratio, "%")

def Teste_S():
    #define the objects of the model 
    S2=Source('S1','Source',interarrivalTime={'distributionType':'Exp','mean':0.5}, entity='Part')
    Q2=Queue('Q1','Queue', capacity=1)
    M2=Machine('M1','Machine', processingTime={'distributionType':'Normal','mean':0.25,'stdev':0.8,'min':0,'max':3})
    E2=Exit('E1','Exit')  

    #define predecessors and successors for the objects    
    S2.defineRouting(successorList=[Q2])
    Q2.defineRouting(predecessorList=[S2],successorList=[M2])
    M2.defineRouting(predecessorList=[Q2],successorList=[E2])
    E2.defineRouting(predecessorList=[M2])

    # call the runSimulation giving the objects and the length of the experiment
    runSimulation(objectList=[S2,Q2,M2,E2], maxSimTime=1440.0, numberOfReplications=5)

    #print the results
    print ("Exits per experiment", E2.Exits)



def two_servers1():

    R=Repairman('R1', 'Bob') 
    S=Source('S1','Source', interarrivalTime={'Exp':{'mean':0.5}}, entity='Part')
    M1=Machine('M1','Machine1', processingTime={'Normal':{'mean':0.25,'stdev':0.1,'min':0.1,'max':1}})
    M2=Machine('M2','Machine2', processingTime={'Normal':{'mean':1.5,'stdev':0.3,'min':0.5,'max':5}})
    Q=Queue('Q1','Queue')
    E=Exit('E1','Exit')  
    #create failures
    F1=Failure(victim=M1, distribution={'TTF':{'Fixed':{'mean':60.0}},'TTR':{'Fixed':{'mean':5.0}}}, repairman=R) 
    F2=Failure(victim=M2, distribution={'TTF':{'Fixed':{'mean':40.0}},'TTR':{'Fixed':{'mean':10.0}}}, repairman=R)

    #define predecessors and successors for the objects    
    S.defineRouting([M1])
    M1.defineRouting([S],[Q])
    Q.defineRouting([M1],[M2])
    M2.defineRouting([Q],[E])
    E.defineRouting([M2])
    
    # add all the objects in a list
    objectList=[S,M1,M2,E,Q,R,F1,F2]  
    # set the length of the experiment  
    maxSimTime=1440.0
    # call the runSimulation giving the objects and the length of the experiment
    runSimulation(objectList, maxSimTime, numberOfReplications=10, seed=1)
        
    print('The exit of each replication is:')
    print(E.Exits)
    
    # calculate confidence interval using the Knowledge Extraction tool
    from KnowledgeExtraction.ConfidenceIntervals import ConfidenceIntervals
    from KnowledgeExtraction.StatisticalMeasures import StatisticalMeasures
    BSM=StatisticalMeasures()
    lb, ub = ConfidenceIntervals().ConfidIntervals(E.Exits, 0.95)
    print('the 95% confidence interval for the throughput is:')
    print('lower bound:', lb)
    print('mean:', BSM.mean(E.Exits))
    print('upper bound:', ub)

def main():
    rngSis = random.SystemRandom()

    R1=Repairman('R1', 'Bob')
    R2=Repairman('R2', 'Joao')

    S=Source('S1','Source', interArrivalTime={'Normal':{'mean':0.5,'stdev':0.25,'min':0.1,'max':1}}, entity='Part')
    M1=Machine('M1','Machine1', processingTime={'Normal':{'mean':0.5,'stdev':0.1,'min':0.1,'max':1}})
    C1 = Conveyer(id='C1', name='M1Q', length= 5, speed= 1.0,capacity = 150)
    Q=Queue(id='Q1',name='Queue',capacity = 150)
    M2=Machine('M2','Machine2', processingTime={'Normal':{'mean':1.0,'stdev':0.3,'min':0.5,'max':5}})
    E=Exit('E1','Exit')  
    #create failures
    F1=Failure(victim=M1, distribution={'TTF':{'Normal':{'mean':60.0,'stdev':10.0,'min':30,'max':70}},
                                        'TTR':{'Normal':{'mean':5.0,'stdev':1.0,'min':2,'max':10}}}, repairman=R1) 
    F2=Failure(victim=M2, distribution={'TTF':{'Normal':{'mean':40.0,'stdev':15.0,'min':25,'max':60}},
                                        'TTR':{'Normal':{'mean':10.0,'stdev':2.0,'min':8,'max':20}}}, repairman=R1)
    #define predecessors and successors for the objects    
    S.defineRouting([M1])
    M1.defineRouting([S],[C1])
    C1.defineRouting([M1],[M2])
    #Q.defineRouting([C1],[M2])
    M2.defineRouting([C1],[E])
    E.defineRouting([M2])
# add all the objects in a list
    objectList=[S,M1,M2,E,C1,R1,F1,F2,R2]  # set the length of the experiment  
    maxSimTime=100.0# call the runSimulation givingthe objects and the length of the experiment
    seed = rngSis.randint(1, 500)
    runSimulation(objectList=objectList, maxSimTime=maxSimTime,numberOfReplications=1,trace='No', seed=seed)# calculate metrics
    blockage_ratio = (M1.totalBlockageTime/maxSimTime)*100
    working_ratio = (R1.totalWorkingTime/maxSimTime)*100# return results for the test
    
    print("parts", E.numOfExits,"blockage_ratio", blockage_ratio,"working_ratio", working_ratio)
#print the results
    print("the system produced", E.numOfExits, "parts")
    print("the blockage ratio of", M1.objName,  "is", blockage_ratio, "%")
    print("the working ratio of",R1.objName,"is", working_ratio, "%")
    print(M1.Get_Status(maxSimTime))
    print(M2.Get_Status(maxSimTime))
    #print(Q.Get_Status(1))
    print('#####')

    M1.Set_processingTime(processingTime={'Normal':{'mean':0.1,'stdev':0.1,'min':0.1,'max':1}})

    maxSimTime=200.0# call the runSimulation givingthe objects and the length of the experiment
    seed = rngSis.randint(1, 500)
    stepSimulation(objectList=objectList, maxSimTime=maxSimTime,numberOfReplications=1,trace='No', seed=seed)# calculate metrics
    blockage_ratio = (M1.totalBlockageTime/maxSimTime)*100
    working_ratio = (R1.totalWorkingTime/maxSimTime)*100# return results for the test

    
    print("parts", E.numOfExits,"blockage_ratio", blockage_ratio,"working_ratio", working_ratio)
#print the results
    print("the system produced", E.numOfExits, "parts")
    print("the blockage ratio of", M1.objName,  "is", blockage_ratio, "%")
    print("the working ratio of",R1.objName,"is", working_ratio, "%")
    st_M1 = M1.Get_Status(maxSimTime)
    print(st_M1)
    print(M2.Get_Status(maxSimTime))
    #print(Q.Get_Status(1))
    print('#####')

    print(st_M1['wp'])


if __name__ == '__main__':
    main()
