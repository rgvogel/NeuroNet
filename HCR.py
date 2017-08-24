import numpy as np
import math
import random
from time import time

def fishFilter(line,length):
    fList = []
    for i in range( 0, length+1):
        #print i
        if i == 0:
            if line[0] == "Strong":
                fList.append(1)
            if line[0] == "Weak":
               fList.append(-1)
        if i == 1:
            if line[i] == "Warm":
                fList.append(1)
            if line[i] == "Moderate":
                fList.append(0)
            if line[i] == "Cold":
                fList.append(-1)
        if i == 2:
            if line[i] == "Warm":
                fList.append(1)
            if line[i] == "Cool":
                fList.append(-1)
        if i == 3:
            if line[i] =="Sunny":
                fList.append(1)
            if line[i] == "Cloudy":
                fList.append(0)
            if line[i] == "Rainy":
                fList.append(-1)
        if i == 4:
            
            x = line[i].strip("\n")
            if x == "Yes":
                
                fList.append(1)
            if x == "No":
                fList.append(-1)
    return fList

def perceptron(inPut, wArray,lvl,start,length, firstLev):
    perSum = 0
    end = 0
    #print " the start: "  + str(start)
    if firstLev == True:
        perSum = 1 * wArray[lvl][start]# bias
        start = start+1
        end = start + length -1
    else:
        end = start + length
    
    count =0
    for x in range(start,end):

        #print "weight A: " + str(start) + str(end)
        perSum =   ((wArray[lvl][x]) *inPut[count]) + perSum
        count +=1
    #print perSum 
    sigma = 1/(1+(math.exp(-perSum)))
    
    return sigma

def startWeightA(F,lev,outNum):
    leng =0 
    att = []
    
    with open(F, 'r') as f:
        for line in f:
            atts = line.split(",")
            length = len(atts) # takes into account bias by including anwser in len
    wLength = length *length  
    weightArray = np.ndarray(shape=(lev,wLength), dtype = float, order = 'C')
    outTot = outNum * length 
    for w in range(0,lev-1):
        for x in range(0,wLength):
            weightArray[w][x] = random.uniform(-.1,.1)
    for c in range(0,outTot):
        weightArray[lev-1][c] = random.uniform(-.1,.1)

    
    return weightArray
def multiPerceptron(atts,weightA,levels,length, hiddenNodes, output, outputNum):
    
    #print atts
    lvWeight = length-1
    firstLevel = True 
    for lv in range (0,levels):
        if lv == levels-1:
            lvWeight = outputNum
            #form(0,.1)
            #print atts
            
        for w in range(0,lvWeight):
            start = w * (length)
            #print "start" + str(start)
            if lv == levels-1:
                output[w] =  perceptron(atts, weightA,lv,start,length, firstLevel)

            else:
                if w == 0:
                    hiddenNodes[lv][w] = 1
                    hiddenNodes[lv][w+1] = perceptron(atts, weightA,lv,start,length, firstLevel)
                    w = w +1
                else:
                    hiddenNodes[lv][w+1] = perceptron(atts, weightA,lv,start,length, firstLevel)

        if lv != levels-1:
            atts = hiddenNodes[lv]
            firstLevel = False
    return hiddenNodes, output

def errorCalc (target, levels, output, outputNum, hiddenNodes, weightA, length, outError, hidNodeError):
    tar = 0
    for o in range (outputNum):
        #print " o and target: " + str(o) +" ," + str(target)
        if o == target:
            tar = 1
        else:
            tar =0
        
        
        outError[o] = (tar- output[o])*output[o]*(1-output[o])
        #print "tar Output: " + str(tar) + ", " + str(output[o]) + ", " + str(outError[o])

    lvWeigh = outputNum
    currentError = outError
    currentSum = 0
    #print outError
    level =0
    #print levels-1
   
    for g in range(length-1):
        currentSum =0
        for w in range(lvWeigh):
                #print g
        
            wgIndex =(g)*(w+1)
            #print str(g)+ ", " + str(w)+ ", " + str(wgIndex)

                #wgIndex =(g)*(w+1)
                
                    
            wgIndex =(g+1)*(w+1)
            cS = outError[w]* weightA[levels-1][wgIndex]    
            currentSum = cS + currentSum
            
            #print weightA
            #print"Error first: " +  str(outError[w]) + ", " + str(weightA[levels-1][wgIndex])+ ", " + str(currentSum) + ", " + str(cS) + ", " + str(g)
          #print "current Error" +str(currentError[w])
                    
            #print "CS: "  + str(currentSum)
            #print hidNodeError
        hidNodeError[0][g] = hiddenNodes[0][g+1]*(1-hiddenNodes[0][g+1])*currentSum
            #print weightA 
            #if g== 0:
             #   print  "hidden Node bias" +str(hiddenNodes[l][g])
              #  print currentSum
            
        currentSum = 0

    return outError, hidNodeError

def weightAdjustment(hiddenNodes, atts,output, weightA, outError, hidNodeError, outputNum, length, levels, teach):

    currentError = outError
    currentNode = output
    lvWeigh = outputNum

    for l in range(0,levels):
        #print "enter"
        #print l
        if l<levels-1:
            #print "yes"
            lvWeigh = length
            currentError= hidNodeError

            if l ==0:
                currentNode = atts
                #print "yes"
            else:
                currentNode = hiddenNodes[l-1]
        wLength = length * (length-1)
        wOut = length *outputNum
        
        if l<levels-1:
            
            for w in range(0,wLength):
                #print currentError 
                #print currentError[l][int(w/length)] 
                errIndex = w/length
                nodeIndex = w
                #print nodeIndex
                outErr =1
                if w> length-1:
                    nodeIndex = w%(length-1)
                
                if nodeIndex >0:
                    outErr = currentNode[nodeIndex-1]
                we = teach*currentError[0][errIndex]*currentNode[nodeIndex]
                bWt = weightA[l][w] 
                #print"error index" +  str(nodeIndex) + str(errIndex)
                weightA[l][w] =  weightA[l][w]+ we
                #print "weight adjustment: " + str(weightA[l][w]) + " ," + str(we) + ", " + " , " + str(bWt) + ", " + str(currentNode[nodeIndex])
        else:
            outIndex =0 
            for o in range(0,wOut):
                #print "hi" + str(o)
                errIndex = o
                outIndex = o/length
                
                #print "outIndex: " + str(outIndex)
                if o> length-1:

                    errIndex = o%(length)
                #print currentError[o]
                
                
                wt = teach*outError[outIndex]*hiddenNodes[0][errIndex]
                #print "err index and outIndex" + str(errIndex)+ ", " + str(outIndex)
                weightA[levels-1][o] =  weightA[levels-1][o]+ wt
                
                #print "weight adjustment: " + str(weightA[levels-1][o]) + ", " + str(teach) + str(outError[outIndex]) + ", " + str(hiddenNodes[0][errIndex])
    return weightA

def neural(File, levels, outputNum, teach, tFile):
    atts = []
    length = 0
    ct =0 
    weightA =startWeightA(File,levels,outputNum)
    #np.array([[1,1,0.5,1,-1,2],[1,1.5,-1,0,0,0]]) #startWeightA(File,levels,outputNum) 
    totError = 1
    totErrorAvg = 1
    totErrorSum = 0
    ct = 0
    Accuracy = 0
    while Accuracy <.9:#>0.00001 or ct>=10:
        totErrorSum= 0
        count = 0
        #print ct
        with open(File, 'r') as fi:    
            count =0
            #for i in range(0,1):
            for line in fi:
                atts =line.split(",")
                #print atts
                if File == "hWTrain.txt":
                    
                    for i in range(0,len(atts)):
                        if i == len(atts) -1:
                            atts[i] = int(atts[i].strip("\n"))
                            #print "hi"
                        else:
                            atts[i] = float(atts[i])/16
                length = len(atts)-1
                wLeng = length * (length-1)
                outT = outputNum * length
                if File == "fish.txt":
                    atts = fishFilter(atts,length)
                #print fishL
                lvWeight = length 
            
                hiddenNodes = np.zeros(shape=((levels-1),length), dtype = np.float64,order = 'F')
                output = np.zeros(shape=(outputNum,), dtype = float, order = 'F')
                hiddenNodes, output = multiPerceptron(atts,weightA,levels,length, hiddenNodes, output, outputNum)
          
                outError = np.zeros(shape =(outputNum,), dtype = float, order = 'F')
                target = atts[length]
                
                hidNodeError = np.zeros(shape =((levels-1),length-1), dtype = float, order = 'F')
                outError, hidNodeError = errorCalc (target, levels,  output, outputNum, hiddenNodes, weightA, length, outError, hidNodeError)
                weightA = weightAdjustment(hiddenNodes, atts,output, weightA, outError, hidNodeError, outputNum, length, levels, teach)
                #for i in range(outputNum): 
                 #   if outputNum >1:
                  #      tIndex = atts[length]-1
                   #     totError =((1-output[tIndex])**2)*.5
                    #else:
                     #   totError =((1-output[0])**2)*.5
                    #totErrorSum = totError + totErrorSum
                #print "Nodes: " + str(atts)
                #print "hidden Nodes: " + str(hiddenNodes)
                #print "H Error: " + str(hidNodeError)
                #print "output: " + str(output)
                #print "O Error: " + str(outError)
                #print weightA
                count+=1
                #print "totError: " + str(totError)
            #totErrorAvg =totErrorSum/(count*outputNum)
            #print "error total avg: " + str(totErrorAvg)
        if ct%10== 0:
            teach = .1
        Accuracy = test(tFile, weightA, outputNum, levels)
        ct +=1

    return weightA 
def test(testData, weighA, outputNum, levels): 
    correct =0
    count = 0
    with open(testData, 'r') as fi:
        
        for line in fi:
            atts =line.split(",")
            #print atts
            length = len(atts)-1
            if testData == "fishTest.txt":
                 atts = fishFilter(atts,length)
            else:
                for i in range(0,len(atts)):
                    if i == len(atts) -1:
                        atts[i] = int(atts[i].strip("\n"))
                    else:
                        atts[i] = float(atts[i])/16

            hiddenNodes = np.zeros(shape=((levels-1),length), dtype = np.float64,order = 'F')
            output = np.zeros(shape=(outputNum,), dtype = float, order = 'F')
            hiddenNodes, output = multiPerceptron(atts,weighA,levels,length, hiddenNodes, output, outputNum) 
            outMax = np.argmax(output)
            count+=1
            #print output
            #print atts
            #print "compared Values: " + str(outMax) + ", " + str(atts[length])
            if outMax == atts[length]:
                
                correct+=1
             
        #print "correct + count: " + str(correct) + ", " + str(count)
        correctPerc = float(correct)/count
    print "percent correct: " + str(correctPerc)
    return correctPerc

if __name__ == "__main__":
    start_time = time()
    levelNum = 2
    outputAmt = 10
    teacher = .5
    F = "hWTrain.txt"
    Ftest = "hWTest.txt"
    weigha = neural(F,levelNum,outputAmt, teacher, Ftest)
    end_time = time()
    time_taken = end_time - start_time
    print "time: " + str(time_taken)
