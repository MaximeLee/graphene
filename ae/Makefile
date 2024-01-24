# Makefile for Fortran project

# Compiler and flags
FC = gfortran
FFLAGS = -Wall -Wextra -std=f2018

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.f90)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.f90,$(OBJ_DIR)/%.o,$(SRC_FILES))

# Executable name
EXEC = $(BIN_DIR)/main

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(FC) $(FFLAGS) $^ -o $@

# Rule to build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	@mkdir -p $(OBJ_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

