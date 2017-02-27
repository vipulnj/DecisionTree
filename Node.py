class Node(object):
    def __init__(self, dataframe):
        self.left = None
        self.attr = None
        self.right = None
        self.isLeaf = None
        self.df = dataframe
        self.label = None
        self.parent = None
        self.possibleLabel = None
        self.serialNumber = None