# Build with: make CADICAL=/path/to/cadical (a CaDiCaL source tree, built with ./configure && make).
CADICAL ?= ../../cadical
CXXFLAGS ?= -O3 -DNDEBUG
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic

lsharp_square: lsharp_square.cpp
	$(CXX) $(CXXFLAGS) -I$(CADICAL)/src -o $@ $< $(CADICAL)/build/libcadical.a

clean:
	rm -f lsharp_square

.PHONY: clean
