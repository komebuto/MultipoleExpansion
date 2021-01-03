CXX       = g++
ifeq ($(PARALLEL), 1)
CXXFLAGS  = -std=c++17 -Wall -g -fopenmp
else
CXXFLAGS  = -std=c++17 -Wall -g
endif
LDFLAGS   = -L/usr/local/lib
LDLIBS    = -lcubature64-1.0.3 -lgsl -lgslcblas -lm
INCLUDE   = -I/usr/local/include -I../include
TARGETS   = program
TARGETDIR = ../bin
SRCROOT   = .
OBJROOT   = ../obj
SRCDIRS  := $(shell find $(SRCROOT) -type d)
SOURCES   = $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))
OBJECTS   = $(addprefix $(OBJROOT)/, $(SOURCES:.cpp=.o))
OBJDIRS   = $(addprefix $(OBJROOT)/, $(SRCDIRS))
DEPENDS   = $(OBJECTS:.o=.d)

PROFILE   = 0
NUMTHREADS= 4

TESTDIR   = ../test
TESTS     = test.cpp
TESTTARGET= testprogram
INPUT     = input.txt 

$(TARGETS): $(OBJECTS) $(LDLIBS)
ifeq ($(PROFILE), 1)
	$(CXX) -pg -o $(TARGETDIR)/$@ $^
else
	$(CXX) -o $(TARGETDIR)/$@ $^
endif

$(OBJROOT)/%.o: $(SRCROOT)/%.cpp
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
ifeq ($(PROFILE), 1)
	$(CXX) -pg $(CXXFLAGS) $(INCLUDE) -o $@ -c $<
else
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<
endif

exec: $(TARGETS)
	$(TARGETDIR)/$<

in: $(TARGETS)
	$(TARGETDIR)/$< < $(TESTDIR)/$(INPUT)

$(TESTTARGET): $(OBJECTS) $(LDLIBS)
ifeq ($(PROFILE), 1)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -pg -o $(TESTDIR)/$@ $(TESTDIR)/$(TESTS) $(filter-out %main.o, $^)
else
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $(TESTDIR)/$@ $(TESTDIR)/$(TESTS) $(filter-out %main.o, $^)
endif

test: $(TESTTARGET)
ifeq ($(PARALLEL), 1)
	export OMP_NUM_THREADS=$(NUMTHREADS)
endif
	$(TESTDIR)/$(TESTTARGET)
ifeq ($(PROFILE), 1)
	gprof $(TESTDIR)/$(TESTTARGET) > $(TESTDIR)/gmon.log
endif


profiletest: $(TESTTARGET)
ifeq ($(PROFILE), 1)
	gprof $(TESTDIR)/$(TESTTARGET) gmon.out
else
	echo "set PROFILE to 1 in makefile"
endif

clean:
	rm -f $(TARGETDIR)/$(TARGETS) $(OBJROOT)/*.o $(TESTDIR)/$(TESTTARGET) *.out *.log $(TESTDIR)/*.log

.PHONY: test, in, dummy
