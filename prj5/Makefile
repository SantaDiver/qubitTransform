VECTOR_FILE = vec.bin
OUTPUT_FILE = outvec.bin
VECTOR_LENGTH = 18

all: prj5 vectorGenerator

%: %.cc
	mpic++ -o $@ $< -m64 -fopenmp

report: all generate
	echo "" > result;
	for processors in 1 2 4; do \
    	mpirun -np $$processors ./prj5 $(VECTOR_LENGTH) 1 $(VECTOR_FILE) $(OUTPUT_FILE) >> result; \
    	mpirun -np $$processors ./prj5 $(VECTOR_LENGTH) 2 $(VECTOR_FILE) $(OUTPUT_FILE) >> result; \
    	mpirun -np $$processors ./prj5 $(VECTOR_LENGTH) 4 $(VECTOR_FILE) $(OUTPUT_FILE) >> result; \
	done
	
generate: vectorGenerator
	mpirun -np 2 ./vectorGenerator $(VECTOR_LENGTH) 2 $(VECTOR_FILE)
	
compare:
	g++ -o vectorComparator vectorComparator.cc -m64
	./vectorComparator $(VECTOR_FILE) $(OUTPUT_FILE)
	rm -rf vectorComparator
	
printin:
	g++ -o vectorReader vectorReader.cc -m64
	./vectorReader $(VECTOR_FILE)
	rm -rf vectorReader
	
printout:
	g++ -o vectorReader vectorReader.cc -m64
	./vectorReader $(OUTPUT_FILE)
	rm -rf vectorReader

clean:
	rm -rf prj5 vectorGenerator *.o
	rm -rf result $(VECTOR_FILE) $(OUTPUT_FILE)
