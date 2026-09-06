from collections import OrderedDict

class LRUCache:
    items: OrderedDict[int, bytes] = OrderedDict()
    max_bytes: int = 100 << 20
    current_bytes: int = 0

    # def __init__(self,size: int):
    #     for i in range(size):
    #         self.items[i] = DecayItem()
    
    # def reinit(self):
    #     self.__init__(len(self.items))

    def item_usable(self, i: int):
        return i in self.items
    
    def process(self):
        if len(self.items) == 0:
            return

        while self.current_bytes > self.max_bytes:
            old_item = self.items.popitem(last=False)[1]
            self.current_bytes -= len(old_item)

    def set_item(self, i: int, item: bytes):
        self.items[i] = item
        self.current_bytes += len(item)

    def get_item(self, i: int):
        if i in self.items:
            self.items.move_to_end(i)
        
        return self.items[i]
    
