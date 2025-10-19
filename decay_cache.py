import gc

class DecayItem:
    item = None
    timer: int = 0

    def __init__(self,
                 item = None,
                 timer: int = 0):
        self.item = item
        self.timer = timer

class DecayCache:
    items: dict[int, DecayItem] = {}
    decay: int = 100

    def __init__(self,size: int):
        for i in range(size):
            self.items[i] = DecayItem()
    
    def reinit(self):
        self.__init__(len(self.items))
    
    def item_decayed(self, i: int):
        return self.items[i].timer == 0

    def item_usable(self, i: int):
        return self.items[i].item != None
    
    def __set_timer(self,i: int,time: int):
        self.items[i].timer = time
    
    def reset_timer(self,i: int):
        self.__set_timer(i,self.decay)
    
    def process(self):
        self.decayed_items = 0
        for i, _ in enumerate(self.items):
            self.items[i].timer = max(0,self.items[i].timer-1)
            
            if self.item_decayed(i):
                self.decayed_items += 1
                self.items[i] = DecayItem()

    def clear(self,ids: list[int]):
        for item_id in ids:
            if self.item_usable(item_id):
                self.reset_timer(item_id)
    
    def set_item(self,i: int,item):
        self.items[i] = DecayItem(item,self.decay)
    
