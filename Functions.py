import math, operator
from Node import Node
import random

count = 0
decisionNodesCount = 0
nodeNumber = 0
nodeNumberVI = 0
totalNumOfDecisionNodesWhileBuildingDT_infoGain = 0
totalNumOfDecisionNodesWhileBuildingDT_varImp = 0

def calculateHighestInfoGain(dataframe, attributesRemaining):
    classresults = dataframe['Class'].tolist() # list of all class results
    numOfClassResults = classresults.__len__() # number of class results

    numOfPosClassResults = classresults.count(1) # number of positive examples
    numOfNegClassResults = classresults.count(0) # number of negative examples

    if numOfClassResults == 0 or numOfPosClassResults == 0 or numOfNegClassResults == 0:
        entropyOfDataSet = 0
    else:
        entropyOfDataSet = - (numOfPosClassResults / numOfClassResults) * math.log2(
            numOfPosClassResults / numOfClassResults) - (numOfNegClassResults / numOfClassResults) * math.log2(
            numOfNegClassResults / numOfClassResults)

    infoGainDict = dict()  # contains attributes and its info gain

    for i in attributesRemaining:
        columni = dataframe[i].tolist() # has the 600 outputs of each of the 20 attributes
        classColumn = dataframe['Class'].tolist() # has 600 outputs, one for each row

        numOfColumniPosInstance = columni.count(1)
        numOfColumniNegInstance = columni.count(0)

        numOfColumniPosInstanceAndPosOutcome = (dataframe[(dataframe[i] == 1) & classColumn == 1]).__len__()
        numOfColumniPosInstanceAndNegOutcome = numOfColumniPosInstance - numOfColumniPosInstanceAndPosOutcome

        numOfColumniNegInstanceAndPosOutcome = (dataframe[(dataframe[i] == 0) & classColumn == 1]).__len__()
        numOfColumniNegInstanceAndNegOutcome = numOfColumniNegInstance - numOfColumniNegInstanceAndPosOutcome

        if numOfColumniPosInstance == 0 or numOfColumniPosInstanceAndPosOutcome == 0 or numOfColumniPosInstanceAndNegOutcome == 0:
            ent1 = 0
        else:
            ent1 = -((numOfColumniPosInstanceAndPosOutcome / numOfColumniPosInstance) * math.log2(
                numOfColumniPosInstanceAndPosOutcome / numOfColumniPosInstance)) - (
                   (numOfColumniPosInstanceAndNegOutcome / numOfColumniPosInstance) * math.log2(
                       numOfColumniPosInstanceAndNegOutcome / numOfColumniPosInstance))

        if numOfColumniNegInstance == 0 or numOfColumniNegInstanceAndPosOutcome == 0 or numOfColumniNegInstanceAndNegOutcome == 0:
            ent2 = 0
        else:
            ent2 = -((numOfColumniNegInstanceAndPosOutcome / numOfColumniNegInstance) * math.log2(
                numOfColumniNegInstanceAndPosOutcome / numOfColumniNegInstance)) - (
                   (numOfColumniNegInstanceAndNegOutcome / numOfColumniNegInstance) * math.log2(
                       numOfColumniNegInstanceAndNegOutcome / numOfColumniNegInstance))

        infoGain = entropyOfDataSet - ((numOfColumniPosInstance / numOfClassResults) * ent1) - (
        (numOfColumniNegInstance / numOfClassResults) * ent2)
        infoGainDict[i] = infoGain

    maxInfoGain = max(infoGainDict.items(), key=operator.itemgetter(1))[0]
    return maxInfoGain


def calculateHighestVarImpurity(dataframe, attributesRemaining):
    classresults = dataframe['Class'].tolist()
    K = classresults.__len__() # total no. of training examples

    K1 = classresults.count(1) # number of training examples with outcome 1
    K0 = classresults.count(0) # number of training examples with outcome 0

    try:
        VIofDataSet = (K0 * K1)/(K^2)
    except ZeroDivisionError:
        VIofDataSet = 0

    varianceImpurityDict = dict()

    for i in attributesRemaining:
        columni = dataframe[i].tolist() # has the 600 outputs of each of the 20 attributes
        classColumn = dataframe['Class'].tolist() # has 600 outputs, one for each row

        numOfColumniPosInstance = columni.count(1) # number of 1s in ith column
        numOfColumniNegInstance = columni.count(0) # number of 0s in ith column

        numOfColumniPosInstanceAndPosOutcome = (dataframe[(dataframe[i] == 1) & classColumn == 1]).__len__() # 1 in ith column and 1 in Class column
        numOfColumniPosInstanceAndNegOutcome = numOfColumniPosInstance - numOfColumniPosInstanceAndPosOutcome # 1 in ith column and 0 in Class column

        numOfColumniNegInstanceAndPosOutcome = (dataframe[(dataframe[i] == 0) & classColumn == 1]).__len__() # 0 in ith column and 1 in Class column
        numOfColumniNegInstanceAndNegOutcome = numOfColumniNegInstance - numOfColumniNegInstanceAndPosOutcome # 0 in ith column and 0 in Class column

        try:
            VI1 = (numOfColumniPosInstanceAndPosOutcome * numOfColumniPosInstanceAndNegOutcome) / ((numOfColumniPosInstance)^2)
        except ZeroDivisionError:
            VI1 = 0

        try:
            VI2 = (numOfColumniNegInstanceAndPosOutcome * numOfColumniNegInstanceAndNegOutcome) / ((numOfColumniNegInstance)^2)
        except ZeroDivisionError:
            VI2 = 0

        varImp = VIofDataSet - ((numOfColumniPosInstance / K1) * VI1) - ((numOfColumniNegInstance / K0) * VI2)
        varianceImpurityDict[i] = varImp

    maxVarImpurity = max(varianceImpurityDict.items(), key=operator.itemgetter(1))[0]
    return maxVarImpurity


def buildDT_infoGain(node):
    global nodeNumber
    global totalNumOfDecisionNodesWhileBuildingDT_infoGain
    classresults = node.df['Class'].tolist() # all values in column Class

    identicalClassResults = True # initially set it to true. In the following loop, if it finds that the any classResults[i] does not match classResults[0], then it means there are no identical values

    for i in classresults:
        if i != classresults[0]: # if any of the subsequent value does not match the first value, then it means there are differing values in class outcomes
            identicalClassResults = False
            break

    if identicalClassResults: # if it is true, find out if it is a set of positive results or negative results
        allResults = classresults[0]
        if allResults == 1:
            node.leaf = True # leaf node
            node.label = 1
            node.attr = None
            return

        else: # allResults = 0
            node.leaf = True # leaf node
            node.label = 0
            node.attr = None
            return

    allColumns = list(node.df) # all columns in dataframe
    attributes = allColumns  # copy array
    attributes.remove('Class') # exclude class as that is the result of attributes

    if(attributes.__len__() == 0):
        # Have run out of attributes. return single node tree with label = most common value of Target attribute
        node.isLeaf = True
        node.attr = None
        numOftimesZeroOccurs = classresults.count(0)
        numOftimesOneOccurs = classresults.count(1)
        if(numOftimesZeroOccurs > numOftimesOneOccurs):
            node.label = 0
        else:
            node.label = 1
        return

    else:
        # There are some attributes remaining
        highestInfoGainAttr = calculateHighestInfoGain(node.df, attributes) # this gets called after dropping the chosen attribute each time. So we will be calculating with a different set of atrributes everytime

        node.attr = highestInfoGainAttr # set the splitting attribute of the node

        zeroValueOnChosenAttribute = node.df[node.df[highestInfoGainAttr] == 0].index.tolist()  # row numbers where the chosen attribute has the value 0
        oneValueOnChosenAttribute = node.df[node.df[highestInfoGainAttr] == 1].index.tolist()  # # row numbers where the chosen attribute has the value 1

        if len(zeroValueOnChosenAttribute) == 0 or len(oneValueOnChosenAttribute) == 0:
            return

        # dataframe with only zero values on chosen attribute. This DF will go on left
        zeroValueRowsOnChosenAttribute_DF = node.df.ix[zeroValueOnChosenAttribute]

        # dataframe with only one values on chosen attribute. This DF will go on right
        oneValueRowsOnChosenAttribute_DF = node.df.ix[oneValueOnChosenAttribute]

        if len(zeroValueOnChosenAttribute) > len(oneValueOnChosenAttribute):
            node.possibleLabel = 0 # used for pruning
        elif len(zeroValueOnChosenAttribute) < len(oneValueOnChosenAttribute):
            node.possibleLabel = 1
        else: # in case both are equal
            node.possibleLabel = 1

        nodeNumber = nodeNumber + 1
        node.serialNumber = nodeNumber
        # numberToAttribute_infoGain[nodeNumber] = highestInfoGainAttr

        totalNumOfDecisionNodesWhileBuildingDT_infoGain = nodeNumber

        node.left = Node(zeroValueRowsOnChosenAttribute_DF) # attach df to left node
        node.right = Node(oneValueRowsOnChosenAttribute_DF)  # attach df to right node

        buildDT_infoGain(node.left) # recursion
        buildDT_infoGain(node.right) # recursion

def buildDT_varImp(node):
    global nodeNumberVI
    global totalNumOfDecisionNodesWhileBuildingDT_varImp
    classresults = node.df['Class'].tolist() # all values in column Class

    identicalClassResults = True # initially set it to true. In the following loop, if it finds that the any classResults[i] does not match classResults[0], then it means there are no identical values

    for i in classresults:
        if i != classresults[0]: # if any of the subsequent value does not match the first value, then it means there are differing values in class outcomes
            identicalClassResults = False
            break

    if identicalClassResults: # if it is true, find out if it is a set of positive results or negative results
        allResults = classresults[0]
        if allResults == 1:
            node.leaf = True # leaf node
            node.label = 1
            node.attr = None
            return
        else: # allResults = 0
            node.leaf = True # leaf node
            node.label = 0
            node.attr = None
            return

    allColumns = list(node.df) # all columns in dataframe
    attributes = allColumns  # copy array
    attributes.remove('Class') # exclude class as that is the result of attributes

    if(attributes.__len__() == 0):
        # Have run out of attributes. return single node tree with label = most common value of Target attribute
        node.isLeaf = True
        node.attr = None
        numOftimesZeroOccurs = classresults.count(0)
        numOftimesOneOccurs = classresults.count(1)
        if numOftimesZeroOccurs > numOftimesOneOccurs:
            node.label = 0
        else:
            node.label = 1
        return

    else:
        # There are some attributes remaining
        highestVarImpAttr = calculateHighestVarImpurity(node.df, attributes) # this gets called after dropping the chosen attribute each time. So we will be calculating with a different set of atrributes everytime

        node.attr = highestVarImpAttr # set the splitting attribute of the node

        zeroValueOnChosenAttribute = node.df[node.df[highestVarImpAttr] == 0].index.tolist()  # row numbers where the chosen attribute has the value 0
        oneValueOnChosenAttribute = node.df[node.df[highestVarImpAttr] == 1].index.tolist()  # # row numbers where the chosen attribute has the value 1

        if len(zeroValueOnChosenAttribute) == 0 or len(oneValueOnChosenAttribute) == 0:
            return

        # dataframe with only zero values on chosen attribute. This DF will go on left
        zeroValueRowsOnChosenAttribute_DF = node.df.ix[zeroValueOnChosenAttribute]

        # dataframe with only one values on chosen attribute. This DF will go on right
        oneValueRowsOnChosenAttribute_DF = node.df.ix[oneValueOnChosenAttribute]

        if len(zeroValueOnChosenAttribute) > len(oneValueOnChosenAttribute):
            node.possibleLabel = 0 # used for pruning
        elif len(zeroValueOnChosenAttribute) < len(oneValueOnChosenAttribute):
            node.possibleLabel = 1
        else: # in case both are equal
            node.possibleLabel = 1

        nodeNumberVI = nodeNumberVI + 1
        node.serialNumber = nodeNumberVI

        totalNumOfDecisionNodesWhileBuildingDT_varImp = nodeNumberVI

        node.left = Node(zeroValueRowsOnChosenAttribute_DF) # attach df to left node
        node.right = Node(oneValueRowsOnChosenAttribute_DF)  # attach df to right node

        buildDT_varImp(node.left) # recursion
        buildDT_varImp(node.right) # recursion


def printTree(node, numofpipes):
    if node is None:
        return

    if not node.isLeaf == True:
        pipestring = ""
        for i in range(numofpipes):
            pipestring = pipestring + "| "
        if (node.left != None):
            if(node.left.label == None):
                outputString = pipestring + node.attr + " = 0"
                print(outputString)
            else:
                outputString = pipestring + node.attr + " = 0 : " + str(node.left.label)
                print(outputString)
            printTree(node.left, numofpipes + 1)
        else:
            return
        if node.right != None:
            if node.right.label == None:
                outputString = pipestring + node.attr + " = 1"
                print(outputString)
            else:
                outputString = pipestring + node.attr + " = 1 : " + str(node.right.label)
                print(outputString)
            printTree(node.right, numofpipes + 1)
        else:
            return


def accuracy(rootnode, df):
    allColumns = list(df)  # all columns in dataframe
    attributes = allColumns  # copy array
    attributes.remove('Class')  # exclude class as that is the result of attributes

    for i in range(len(df)): # for every row of the DF -- len(df)
        rowValuesDict = dict() # fresh empty dictionary for each row
        for j in attributes:
            value = df.loc[i,j]
            rowValuesDict[j] = value # add to the dictionary
        testSetValue = df.loc[i, "Class"]
        isCorrectClassification(rowValuesDict, rootnode, testSetValue)
    acc = count/len(df) * 100
    resetCount()
    return acc

def isCorrectClassification(dict1, rootnode, testSetValue):
    global count
    try:
        valueOnTestSet = dict1[rootnode.attr]
    except KeyError:
        if rootnode.label == testSetValue: # Both values match...
            count = count + 1
            return
        else: # values don't match
            return

    if valueOnTestSet == 0:
        try:
            if not rootnode.left.isLeaf:
                isCorrectClassification(dict1, rootnode.left,testSetValue) # recursion
        except AttributeError:
            return
    elif valueOnTestSet == 1:
        try:
            if not rootnode.right.isLeaf:
                isCorrectClassification(dict1, rootnode.right, testSetValue) # recursion
        except AttributeError:
            return

def resetCount():
    global count
    count = 0
    return


def pruneTree(Lvalue, Kvalue, rootnode, valdf):
    global decisionNodesCount
    rootnodeCopy = rootnode
    accTreeDict = dict()
    # accuracies = []
    for i in range(int(Lvalue)):
        Mvalue = random.randrange(1, int(Kvalue))
        numOfDecisionNodes = totalNumOfDecisionNodesWhileBuildingDT_infoGain
        accuracies = []
        for i in range(Mvalue):
            decisionNodesCount = 0 # reset count
        Pvalue = random.randrange(1, numOfDecisionNodes)
        chopNodes(Pvalue, rootnodeCopy) # rootnodecopy will be chopped. 'L' number of decision trees to be created and checked
        accval = accuracy(rootnodeCopy, valdf)
        accTreeDict[accval] = rootnodeCopy
    return accTreeDict

def chopNodes(Pvalue, rootnode):

    if rootnode.isLeaf:
        return
    directions = []
    for i in range(Pvalue):
        LR = ['left', 'right', 'left', 'right', 'left', 'right']
        directions.append(random.choice(LR))

    if directions.__len__() > 12:
        cut = random.randrange(5,12)
        directions= directions[:cut]

    for dir in directions:
        if dir == "left":
            if rootnode.left.left is not None or rootnode.left.right is not None:
                rootnode = rootnode.left
        if dir == "right":
            if rootnode.right.left is not None or rootnode.right.right is not None:
                rootnode = rootnode.right

    # prune the sub tree
    rootnode.left = None
    rootnode.right = None
    rootnode.isLeaf = True
    rootnode.label = rootnode.possibleLabel