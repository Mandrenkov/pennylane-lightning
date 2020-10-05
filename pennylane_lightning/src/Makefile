# Compile with minimum number of compiler flags, hopefully taking the least amount of time.
dev:
	g++ -fPIC -shared -O3 `python3 -m pybind11 --includes` tensor.cpp utils.cpp stopwatch.cpp lightning_qubit.cpp -o lightning_qubit.so -lopenblas
	#g++ -shared -std=c++14 -fPIC `python3 -m pybind11 --includes` lightning_qubit.cpp -o lightning_qubit_ops.so -lopenblas -lblas -llapack -lgfortran
