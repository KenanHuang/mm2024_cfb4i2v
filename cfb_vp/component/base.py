class heapQ():
    def __init__(self, cap, mode='min'):
        self.queue = [None] * (cap + 1)  
        self.maxCap = cap
        self.cap = 0
        self.mode = mode
        assert self.mode in ['max', 'min']  # Maximum heap or minimum heap

    def getPureData(self):
        rcQueue = self.queue[1:self.cap + 1].copy()
        return [data[1] for idx, data in enumerate(rcQueue) if data is not None] if self.cap > 0 else []

    def left(self, idx):  
        return idx * 2

    def right(self, idx):  
        return self.left(idx) + 1

    def parent(self, idx):  
        return idx // 2

    def compareVal(self, idx_i, idx_j):
        if self.mode == 'min':
            return self.queue[idx_i][0] < self.queue[idx_j][0]
        if self.mode == 'max':
            return self.queue[idx_i][0] > self.queue[idx_j][0]

    def swap(self, idx_i, idx_j):
        tmp = self.queue[idx_i]
        self.queue[idx_i] = self.queue[idx_j]
        self.queue[idx_j] = tmp

    def swim(self, idx): 
        while idx > 1: 
            parIdx = self.parent(idx)
            if self.compareVal(idx, parIdx):
                self.swap(idx, parIdx)
            idx = parIdx

    def sink(self, idx):  
        while self.left(idx) <= self.cap: 
            # get Max ChildIdx
            maxChildIdx = self.left(idx)
            rightChildIdx = self.right(idx)
            if rightChildIdx <= self.cap and self.compareVal(rightChildIdx, maxChildIdx):
                maxChildIdx = rightChildIdx

            if self.compareVal(idx, maxChildIdx): break  # par is larger than any child no need to sink
            self.swap(maxChildIdx, idx)

            idx = maxChildIdx

    def insert(self, prob, val):  
        if self.cap < self.maxCap:
            self.cap += 1
            self.queue[self.cap] = (prob, val)
            self.swim(self.cap)
        else:
            p, v = self.getTop()
            if p < prob:
                self.deleteMax()
                self.insert(prob, val)

    def deleteMax(self):  
        topIdx = 1
        maxVal = self.queue[topIdx]
        self.swap(topIdx, self.cap)  # swap the top and the tail
        self.cap -= 1
        self.sink(topIdx)
        return maxVal

    def getTop(self):
        topIdx = 1
        maxVal = self.queue[topIdx]
        return maxVal


if __name__ == '__main__':
    arr = heapQ(cap=10)
    for i in range(10):
        arr.insert(10 - i, i)

    print(1)
