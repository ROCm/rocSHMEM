#!/bin/bash

# Function to display help information
function display_help {
    echo "Usage:"
    echo "  $0 binary_name all                     # Runs all standard tests"
    echo "  $0 binary_name custom <ranks> <filter> # Runs custom test configuration"
    echo
    echo "Arguments:"
    echo "  binary_name: Name of the binary to run."
    echo "  all: Executes predefined test configurations."
    echo "  custom: Executes a test with custom MPI ranks and GTest filter."
    echo "  ranks: Number of MPI ranks (required for custom mode)."
    echo "  filter: GTest filter string (required for custom mode)."
    echo
}

# Validate number of arguments for each mode
if [[ "$#" -lt 2 ]] || 
   { [[ "$2" == "all" ]] && [[ "$#" -ne 2 ]]; } ||
   { [[ "$2" == "custom" ]] && [[ "$#" -ne 4 ]]; }; then
    display_help
    exit 1
fi

binary_name=$1
mode=$2
timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
log_file="unit_tests_${timestamp}.log"

# Function to execute mpirun command
function run_mpirun {
    local np=$1
    local gtest_filter=$2
    echo "mpirun -np $np $binary_name --gtest_filter='$gtest_filter'" | tee -a "$log_file"
    mpirun -np "$np" "$binary_name" --gtest_filter="$gtest_filter" >> "$log_file" 2>&1
}

# Processing modes
case $mode in
    all)
        run_mpirun 4 "-IPCImplSimpleCoarseTestFixture.*:IPCImplSimpleFineTestFixture.*"
        run_mpirun 2 "IPCImplSimpleCoarseTestFixture.*:IPCImplSimpleFineTestFixture.*"
        ;;
    custom)
        # Check if ranks is a positive integer
        if [[ "$3" -le 1 ]]; then
            echo "Error: 'ranks' must be a positive integer."
            display_help
            exit 1
        fi
        run_mpirun $3 $4
        ;;
    *)
        echo "Error: Invalid mode '$mode'." | tee -a "$log_file"
        display_help
        exit 1
        ;;
esac

echo "Tests Completed"
echo "log file: '$log_file'"
