all: prj4

%: %.cc
	mpic++ -o $@ $< -m64 -fopenmp

report: all
	echo "" > result;
	for cube in 23; do \
    	mpirun -np 1 ./prj4 $$cube 7 17 1 >> result; \
    	mpirun -np 1 ./prj4 $$cube 7 17 2 >> result; \
    	mpirun -np 2 ./prj4 $$cube 7 17 1 >> result; \
    	mpirun -np 2 ./prj4 $$cube 7 17 2 >> result; \
	done

clean:
	rm -rf prj4 *.o
