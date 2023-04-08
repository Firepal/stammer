import importlib.util
import subprocess

has_modules = True
with open("requirements.txt","r") as modules:
    for line in modules.readlines():
        check_name = line[:-1]
        if check_name == '': continue
        if check_name == "Pillow":
            check_name = "PIL"
        
        if importlib.util.find_spec(check_name) is None:
            print(f"need to fetch {check_name}")
            has_modules = False
            break

if not has_modules:
    print("Fetching Python required modules for STAMMER\n")
    proc = subprocess.Popen(
        ['pip','install','-r','requirements.txt']
    )
    proc.communicate()
    if proc.returncode == 0:
        print("Modules fetched.")

from gui_bits import StammerGUI

tks = StammerGUI()
tks.mainloop()