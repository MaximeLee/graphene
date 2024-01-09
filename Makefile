activate:
	source ./env/bin/activate

compile_parallel:
	python compile.py True False

compile_serial:
	python compile.py False False

compile_debug:
	python compile.py False True

# clean .c .so .html files
clean:
	find graphene -name "*.c" -type f -delete
	find graphene -name "*.cpp" -type f -delete
	find graphene -name "*.so" -type f -delete
	find graphene -name "*.html" -type f -delete
	rm -f *.so
	rm -rf cython_debug build
