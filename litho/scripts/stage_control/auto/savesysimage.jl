using Pkg
ENV["PYTHON"] = abspath("C:\\Users\\Hacker Fab\\AppData\\Local\\Programs\\Python\\Python310\\python.exe")
Pkg.build("PyCall")
using PackageCompiler
PackageCompiler.create_sysimage(; sysimage_path="JuliaSysimage.so",
                                     precompile_execution_file="stage_simulate.jl")