# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#!/bin/bash

echo Test Name $2

INPUTS=/mnt/mlebeane/spts_data

case $2 in
    *"single_thread")
        mpirun -np 2 $1 -f $INPUTS/test_matrices/diagonal_large.mtx -a 2 -b 512 -p 64 -v -i 3 > $3/diagonal_large_bput.log
        mpirun -np 2 $1 -f $INPUTS/test_matrices/not_quite_diagonal.mtx -a 2 -b 256 -p 64 -v -i 3 > $3/not_quite_diagonal_bput.log
        ;;
    *"multi_thread")
        mpirun -np 2 $1 -f $INPUTS/test_matrices/diagonal_large.mtx -a 2 -b 512 -p 64 -v -i 3 > $3/diagonal_large_bput.log
        mpirun -np 2 $1 -f $INPUTS/test_matrices/not_quite_diagonal.mtx -a 2 -b 256 -p 64 -v -i 3 > $3/not_quite_diagonal_bput.log
        mpirun -np 2 $1 -f $INPUTS/test_matrices/not_quite_diagonal.mtx -a 1 -b 256 -v -i 3 > $3/not_quite_diagonal_get.log
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $?
