all: prj1

%: %.cc
	mpic++ -o $@ $< -m64

report: all
	echo "" > result;
	for cube in 19 20 21; do \
    	mpirun -np 1 ./prj1 $$cube 1 >> result; \
    	mpirun -np 2 ./prj1 $$cube 1 >> result; \
    	mpirun -np 4 ./prj1 $$cube 1 >> result; \
    	mpirun -np 8 ./prj1 $$cube 1 >> result; \
	done
	
clean:
	rm -rf prj1 *.o
