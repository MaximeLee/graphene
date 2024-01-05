compile:
	python compile.py

# clean .c .so .html files
clean:
	find . -name "*.c" -type f -delete
	find . -name "*.so" -type f -delete
	find . -name "*.html" -type f -delete
