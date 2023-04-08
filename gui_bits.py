import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
import stammer
from pathlib import Path
from threading import Thread
import json
from typing import Callable, Any

RECENT_FILENAME = ".stammer_recent"

def get_recent_path():
    return Path.home() / RECENT_FILENAME

def write_recent_file(data: dict):
    dest_fp = open(get_recent_path(),"wt+")
    json.dump(data,dest_fp,indent=4,separators=(',',':'))
    dest_fp.close()

def parse_recent_file():
    if not get_recent_path().exists: return None
    return json.load(get_recent_path())


class Picker(tk.Frame):
    def __init__(self,*args,label_text = "",entry_text = "",**kwargs):
        super().__init__(*args,**kwargs)
        lbl = tk.Label(self,text=label_text)
        lbl.grid(column=0,row=0)
        
        btn_pick = ttk.Button(self,text=self.get_pick_text(),command=self.ask_user)
        btn_pick.grid(column=1,row=0,padx=12)

        self.entry = tk.Entry(self)
        self.entry.grid(column=2,row=0,ipadx=128)
        self.set_entry_text(str(entry_text))

        self.callbacks = []

    def ask_user_function(self) -> Path:
        raise NotImplementedError
    
    def do_callbacks(self,path: Path):
        for cb in self.callbacks:
            cb(path)

    def ask_user(self):
        path = self.ask_user_function()
        print(path)
        if path != None or path != ".":
            self.set_entry_text(str(path))
            self.do_callbacks(path)
    
    def get_pick_text(self):
        return "Pick..."
    
    def set_entry_text(self,text: str):
        self.entry.delete(0,tk.END)
        self.entry.insert(0,text)
    
    def get_path(self):
        return Path(self.entry.get()).resolve()

    def add_callback_picked(self, callback: Callable[[Path],Any]):
        self.callbacks.append(callback)

class FilePicker(Picker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def ask_user_function(self) -> Path:
        return Path(tkfd.askopenfilename())

class TopMenu(tk.Menu):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        recent = tk.Menu(self)
        self.add_cascade(label="Recent",menu=recent)

        self.carrier = tk.Menu(recent)
        
        self.modulator = tk.Menu(recent)

        recent.add_cascade(label="Carrier",menu=self.carrier)
        recent.add_cascade(label="Modulator",menu=self.modulator)
    
    def menu_items(menu: tk.Menu):
        items = []
        last_item_idx = menu.index('end')
        for item in range(last_item_idx+1):
            items.append(menu.entrycget(item,'label'))
        return items
    
    def set_items(items: list[str], menu: tk.Menu):
        menu.delete(0,'end')
        def ye():
            pass
        for i in items:
            menu.add_command(label=i,command=ye)

    def add_path_to(self,menu, path: Path):
        menu.add_command(label=str(path))

    def add_recent_carrier_path(self,path: Path):
        self.add_path_to(self.carrier,path)

    def save_recent(self):
        data_dict = {}
        for i in self.menu_items(self.carrier):
            print(i)

class StammerGUI(tk.Tk):
    help_text = "\
Pick an audio or video file for carrier and modulator.\n\
The carrier will be edited to resemble the modulator."
    stammer_thread: Thread
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # remove grey tearoff line
        self.option_add('*tearOff',False)

        menu = TopMenu(self)

        self.config(menu=menu)

        lbl = tk.Label(self,text=self.help_text)
        lbl.pack(side="top")

        dp_frame = tk.Frame(self)

        def make_filepicker(text,entry_text):
            x = FilePicker(dp_frame,label_text=text,entry_text=entry_text)
            x.grid(sticky="e")
            
            return x
        
        fp_carrier = make_filepicker("Carrier:","")
        fp_carrier.add_callback_picked(menu.add_recent_carrier_path)

        fp_mod = make_filepicker("Modulator:","")
        fp_output = make_filepicker("Output file:","")

        dp_frame.pack(padx=12,pady=12)

        def boop():
            print(f"carrier: {fp_carrier.get_path()}")
            print(f"modulator: {fp_mod.get_path()}")

            def worker(*args):
                stammer.process(*args)
                print("complete")
            
            args = (
                fp_carrier.get_path(),
                fp_mod.get_path(),
                fp_output.get_path(),
                None,
                "basic",
                "mem_decay",
                "full",
                1
            )
            
            self.stammer_thread = Thread(target=worker,args=args)
            self.stammer_thread.start()

        start_btn = ttk.Button(self,text="Start",command=boop)
        start_btn.pack(side="bottom")
