
using ZMQ
using ZeroMQ_jll
using GLMakie
GLMakie.activate!(inline=false)


ctx = Context()
amscope = Socket(ctx, SUB)
# Set Conflate == 1
rc = ccall((:zmq_setsockopt, libzmq), Cint, (Ptr{Cvoid}, Cint, Ref{Cint}, Csize_t), amscope, 54, 1, sizeof(Cint))
ZMQ.subscribe(amscope, "")
connect(amscope, "tcp://10.193.10.1:5556")



dd = ZMQ.recv(amscope)
size(dd)[1] / (1824 * 1216)