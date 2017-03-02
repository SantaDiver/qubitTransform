all: prj1

%: %.cc
	g++ -std=c++11 $< -o $@ -fopenmp

report: all
	echo "" > result;
	for cube in 20 24 26; do \
    	./prj1 1 $$cube -adamar 1 >> result; \
    	./prj1 2 $$cube -adamar 1 >> result; \
    	./prj1 4 $$cube -adamar 1 >> result; \
    	./prj1 8 $$cube -adamar 1 >> result; \
	done
