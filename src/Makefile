PARALLEL  = 1
NUMTHREADS= 36

CXX       = g++
ifeq ($(PARALLEL), 1)
CXXFLAGS  = -std=c++14 -Wall -g -fopenmp
else
CXXFLAGS  = -std=c++14 -Wall -g
endif
LDFLAGS   = -L/usr/local/lib
LDLIBS    = -lcubature64-1.0.3 -lgsl -lgslcblas -lm -lpython3.8 -lcomplex_bessel
INCLUDE   = -I/usr/local/include -I../include -I/usr/include/python3.8
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

TESTDIR   = ../test
TESTS     = test.cpp
TESTOUTDIR= ../test/out
TESTTARGET= testprogram
INPUT     = input.txt 

# makes directory of name $(1) if not existed
Makedir   = @if [ ! -e $(1) ]; then mkdir -p $(1); fi 

$(TARGETS): $(OBJECTS) $(LDLIBS)
	$(call Makedir, $(TARGETDIR))
ifeq ($(PROFILE), 1)
	$(CXX) -pg -o $(TARGETDIR)/$@ $^
else
	$(CXX) -o $(TARGETDIR)/$@ $^
endif

$(OBJROOT)/%.o: $(SRCROOT)/%.cpp
	$(call Makedir, `dirname $@`)
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
	$(call Makedir, $(TESTOUTDIR))
ifeq ($(PROFILE), 1)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -pg -o $(TESTOUTDIR)/$@ $(TESTDIR)/$(TESTS) $(filter-out %main.o, $^)
else
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $(TESTOUTDIR)/$@ $(TESTDIR)/$(TESTS) $(filter-out %main.o, $^)
endif

test: $(TESTTARGET)
ifeq ($(PARALLEL), 1)
	export OMP_NUM_THREADS=$(NUMTHREADS)
endif
	$(TESTOUTDIR)/$(TESTTARGET)
ifeq ($(PROFILE), 1)
	gprof $(TESTOUTDIR)/$(TESTTARGET) > $(TESTOUTDIR)/gmon.log
endif


profiletest: $(TESTTARGET)
ifeq ($(PROFILE), 1)
	gprof $(TESTOUTDIR)/$(TESTTARGET) gmon.out
else
	echo "set PROFILE to 1 in makefile"
endif

clean:
	rm -f $(TESTOUTDIR)/* $(TESTDIR)/*tmp* $(OBJROOT)/*.o $(TARGETDIR)/*

.PHONY: test, in, dummy
