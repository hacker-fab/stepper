# using Pkg
# ENV["PYTHON"] = abspath("venv/bin/python")
# Pkg.build("PyCall")
using PyCall
using Observables

const PYLOCK = Ref{ReentrantLock}()
PYLOCK[] = ReentrantLock()

# acquire the lock before any code calls Python
pylock(f::Function) =
    Base.lock(PYLOCK[]) do
        prev_gc = GC.enable(false)
        try
            return f()
        finally
            GC.enable(prev_gc) # recover previous state
        end
    end

# the file "vision_flir.py" had to be removed from the repository 
@pyinclude("vision_flir.py")
liveimgsz = Int[1368, 912]
# liveimgsz = Int[2736, 1824]
# @pyinclude("vision_v4l2.py")
# liveimgsz = Int[1920, 1080]
@pyinclude("align.py")

py"""
import numpy as np
def preallocarr_live():
    return np.zeros($liveimgsz, dtype = np.uint8)
"""

liveimg = Observable(pycall(py"""preallocarr_live""", PyArray))
annoimg = Observable(pycall(py"""preallocarr_live""", PyArray))
mouseimg = Observable(pycall(py"""preallocarr_live""", PyArray))

visinit = true
function vislooponce(visionCh, refCh)
    global running, liveimg, visinit, shiftimgcrop, originxy
    py"""get_img($(liveimg[]))"""
    if visinit
        margin = 75
        cropx, cropy = size(liveimg[], 1)÷2-margin:size(liveimg[], 1)÷2+margin,
        size(liveimg[], 2)÷2-margin:size(liveimg[], 2)÷2+margin
        shiftx, shifty = cropx, cropy
        shiftimgcrop = liveimg[][cropx, cropy]
        originxy = [shiftx[1], shifty[1]]
        visinit = false
    end
    if isready(refCh)
        xoff, yoff = take!(refCh)

        # Visualization without margin
        margin = 0
        cropx, cropy = max(1 + margin, 1 - xoff):min(liveimgsz[1] - margin, liveimgsz[1] - xoff),
        max(1 + margin, 1 - yoff):min(liveimgsz[2] - margin, liveimgsz[2] - yoff)
        shiftx, shifty = max(1, 1 + xoff):min(liveimgsz[1], liveimgsz[1] + xoff),
        max(1, 1 + yoff):min(liveimgsz[2], liveimgsz[2] + yoff)

        mouseimg[] .= 0.0
        mouseimg[][shiftx, shifty] .= liveimg[][cropx, cropy]

        # Template Matching with margin
        # marginx = 1200
        # marginy = 700
        marginx = 600
        marginy = 400
        cropx, cropy = max(1 + marginx, 1 - xoff):min(liveimgsz[1] - marginx, liveimgsz[1] - xoff),
        max(1 + marginy, 1 - yoff):min(liveimgsz[2] - marginy, liveimgsz[2] - yoff)
        if length(cropx) <= 10 || length(cropy) <= 10
            return
        end
        shiftimgcrop = liveimg[][cropx, cropy]
        originxy = [xoff + cropx[1], yoff + cropy[1]]
    end
    dxy = py"""align($(liveimg[]), $(shiftimgcrop), $(annoimg[]))"""
    dxy .+= 1 # convert to julia indexing

    put!(visionCh, [Int64(time_ns()), Int64.([originxy[1] - dxy[1], originxy[2] - dxy[2]])..., Int64(0)])
end