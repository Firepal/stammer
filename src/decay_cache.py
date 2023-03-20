import numpy as np
from scipy.sparse import dok_array
import time
import random

class Frame:
    frame: bytes = None
    timer: int = 0

    def __init__(self,
                 frame: bytes = None,
                 timer: int = 0):
        self.frame = frame
        self.timer = timer

class DecayArray:
    array: dict[Frame] = {}
    timer_val: int = 100
    requested_frames: set[int] = []
    bad_frames = 0

    def __init__(self,size: int):
        self.timer_val = size / 4
        for i in range(size):
            self.array[i] = Frame(None, 0)
    
    def item_decayed(self, i: int):
        return self.array[i].timer == 0

    def item_usable(self, i: int):
        return self.array[i].frame != None
    
    def __set_timer(self,i: int,time: int):
        self.array[i].timer = time
    
    def reset_timer(self,i: int):
        self.__set_timer(i,self.timer_val)
    
    def process(self):
        self.bad_frames = 0
        for i in range(len(self.array)):
            # already_decayed = self.item_decayed(i)

            self.array[i].timer = max(0,self.array[i].timer-1)
            
            if self.item_decayed(i):
                self.bad_frames += 1
                self.array[i].frame = None
    
    def used_frames(self,frame_ids: list[int]):
        for id in frame_ids:
            maxsize = len(self.array.keys())
            for i in range(id, min(id+1000,maxsize)):
                # if self.item_decayed(i):
                #     self.requested_frames.add(i)
                self.reset_timer(i)
                # if i == id: print("used", str(i))
    
    def clear_requested(self,frame_ids: list[int]):
        for id in frame_ids:
            self.reset_timer(id)
    
    def set_frame(self,frame: bytes, i: int):
        self.array[i] = Frame(frame,self.timer_val)
    
