import sys
import Functions as func
import pandas as pd
from Node import Node

if __name__ == "__main__":
    Lvalue = sys.argv[1]
    Kvalue = sys.argv[2]
    tr_setFile = sys.argv[3]
    val_setFile = sys.argv[4]
    test_setFile = sys.argv[5]
    shouldPrint = sys.argv[6]

    tr_set = pd.read_csv(tr_setFile)
    test_set = pd.read_csv(test_setFile)
    val_set = pd.read_csv(val_setFile)

    if isinstance(tr_set, pd.DataFrame): # check if the variable is dataframe
        infoGainTreeRoot = Node(tr_set) # attach the dataset to the node
        func.buildDT_infoGain(infoGainTreeRoot)

        varImpTreeRoot = Node(tr_set) # attach the dataset to the node
        func.buildDT_varImp(varImpTreeRoot)
    else:
        print("Data did not scan successfully.")
        exit()

    infoGainBuiltTreeRoot = infoGainTreeRoot # now the trees are fully built
    varImpBuiltTreeRoot = varImpTreeRoot

    if (shouldPrint == "yes"):
        print("********************* Information Gain Hueristic (unpruned tree) **********\n\n")
        func.printTree(infoGainBuiltTreeRoot, 0)
        print("\n\n\********************* Variance Impurity Hueristic (unpruned tree) **********\n\n")
        func.printTree(varImpBuiltTreeRoot, 0)

    accuracy_infoGain = func.accuracy(infoGainBuiltTreeRoot, test_set)
    print("Accuracy for tree built using Information Gain against test set (without pruning) = ", accuracy_infoGain)

    accuracy_varImp = func.accuracy(varImpBuiltTreeRoot, test_set)
    print("Accuracy for tree built using Variance Impurity against test set (without pruning) =", accuracy_varImp)

    print("------------ PRUNING ------------")
    pruneResult_infoGain = func.pruneTree(Lvalue, Kvalue, infoGainBuiltTreeRoot, val_set)
    maxResult_infoGain = max(pruneResult_infoGain.keys())
    if maxResult_infoGain > accuracy_infoGain:
        finalAcc_infoGain = maxResult_infoGain
        if shouldPrint == "yes":
            print("The pruned tree for Information Gain has better results. The tree is as follows:")
            prunedTreeRootNode_IG = pruneResult_infoGain[finalAcc_infoGain]
            func.printTree(prunedTreeRootNode_IG, 0)
    else:
        finalAcc_infoGain = accuracy_infoGain
    print("Accuracy for tree built using Information Gain against validation set (with pruning) = ", finalAcc_infoGain)

    pruneResult_varImp = func.pruneTree(Lvalue, Kvalue, varImpBuiltTreeRoot, val_set)
    maxResult_varImp = max(pruneResult_varImp.keys())
    if maxResult_varImp > accuracy_varImp:
        finalAcc_varImp = maxResult_varImp
        if shouldPrint == "yes":
            print("The pruned tree for Variance Impurity has better results. The tree is as follows:")
            prunedTreeRootNode_VI = pruneResult_varImp[finalAcc_varImp]
            func.printTree(prunedTreeRootNode_VI, 0)
    else:
        finalAcc_varImp = accuracy_varImp
    print("Accuracy for tree built using Information Gain against validation set (with pruning) = ", finalAcc_varImp)