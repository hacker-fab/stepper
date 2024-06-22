## UPDATE: 
A few Flir-based files had to be removed, and as such, the steps below no longer work (but they are still left here for reference). If you are a member of CMU Hacker Fab and require these files, search for "flir" or "pyspin" in the associated archived repository to find commits with the files.

### Steps to run
- Make sure *BOTH* Arduino, Camera and Motor power are connected
- run `source setup.bash`
- run `python vision_flir_reset.py`
- wait 10 seconds
- run `sudo ./setup.bash`
- run `python vision_flir_setup.py`
- run  C:\\julia\\bin\\julia --project=. --sysimage JuliaSysimage.so main.jl