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

driver_return_status=0

if [ $# -eq 0 ] ; then
    echo "This script must be run with at least 2 arguments."
    echo 'Usage: ${0} argument1 argument2 [argument3]'
    echo "  argument1 : path to the tester driver"
    echo "  argument2 : test type to run, e.g put"
    echo "  argument3 : directory to put the output logs"
    exit 1
fi

echo "Test Name ${2}"

check() {
    if [ $? -ne 0 ]
    then
        echo "Failed $1" >&2
        driver_return_status=1
    fi
}

case $2 in
    ###########################################################################
    ############################## SERIAL TESTS ###############################
    ###########################################################################
    *"serial")
        echo "get_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 0 > $3/get_n2_w1_z1_1MB.log
        check get_n2_w1_z1_1MB
        echo "getnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 1 > $3/getnbi_n2_w1_z1_1MB.log
        check getnbi_n2_w1_z1_1MB
        echo "put_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 2 > $3/put_n2_w1_z1_1MB.log
        check put_n2_w1_z1_1MB
        echo "putnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 3 > $3/putnbi_n2_w1_z1_1MB.log
        check putnbi_n2_w1_z1_1MB
        echo "wg_get_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 25 > $3/wg_get_n2_w1_z64_1MB.log
        check wg_get_n2_w1_z1_1MB
        echo "wg_getnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 26 > $3/wg_getnbi_n2_w1_z64_1MB.log
        check wg_getnbi_n2_w1_z1_1MB
        echo "wg_put_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 27 > $3/wg_put_n2_w1_z64_1MB.log
        check wg_put_n2_w1_z1_1MB
        echo "wg_putnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 28 > $3/wg_putnbi_n2_w1_z64_1MB.log
        check wg_putnbi_n2_w1_z1_1MB
        echo "wg_get_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 64 -s 1048576 -a 25 > $3/wg_get_tiled_n2_w2_z64_1MB.log
        check wg_get_tiled_n2_w1_z1_1MB
        echo "wg_getnbi_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 64 -s 1048576 -a 26 > $3/wg_getnbi_tiled_n2_w2_z64_1MB.log
        check wg_getnbi_tiled_n2_w1_z1_1MB
        echo "wg_put_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 64 -s 1048576 -a 27 > $3/wg_put_tiled_n2_w2_z64_1MB.log
        check wg_put_tiled_n2_w1_z1_1MB
        echo "wg_putnbi_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 64 -s 1048576 -a 28 > $3/wg_putnbi_tiled_n2_w2_z64_1MB.log
        check wg_putnbi_tiled_n2_w1_z1_1MB
        echo "wave_get_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 29 > $3/wave_get_n2_w1_z64_1MB.log
        check wave_get_n2_w1_z1_1MB
        echo "wave_getnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 30 > $3/wave_getnbi_n2_w1_z64_1MB.log
        check wave_getnbi_n2_w1_z1_1MB
        echo "wave_put_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 31 > $3/wave_put_n2_w1_z64_1MB.log
        check wave_put_n2_w1_z1_1MB
        echo "wave_putnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 1048576 -a 32 > $3/wave_putnbi_n2_w1_z64_1MB.log
        check wave_putnbi_n2_w1_z1_1MB
        echo "wave_get_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 128 -s 1048576 -a 29 > $3/wave_get_tiled_n2_w2_z128_1MB.log
        check wave_get_tiled_n2_w1_z1_1MB
        echo "wave_getnbi_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 128 -s 1048576 -a 30 > $3/wave_getnbi_tiled_n2_w2_z128_1MB.log
        check wave_getnbi_tiled_n2_w1_z1_1MB
        echo "wave_put_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 128 -s 1048576 -a 31 > $3/wave_put_tiled_n2_w2_z128_1MB.log
        check wave_put_tiled_n2_w1_z1_1MB
        echo "wave_putnbi_tiled_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 128 -s 1048576 -a 32 > $3/wave_putnbi_tiled_n2_w2_z128_1MB.log
        check wave_putnbi_tiled_n2_w1_z1_1MB
        echo "amofadd_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 5 > $3/amofadd_n2_w1_z1.log
        check amofadd_n2_w1_z1
        echo "amofinc_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 6 > $3/amofinc_n2_w1_z1.log
        check amofinc_n2_w1_z1
        echo "amofetch_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 7 > $3/amofetch_n2_w1_z1.log
        check amofetch_n2_w1_z1
        echo "amofcswap_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 8 > $3/amofcswap_n2_w1_z1.log
        check amofcswap_n2_w1_z1
        echo "amoadd_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 9 > $3/amoadd_n2_w1_z1.log
        check amoadd_n2_w1_z1
        echo "amoinc_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 10 > $3/amoinc_n2_w1_z1.log
        check amoinc_n2_w1_z1
        echo "pingpong_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 13 > $3/pingpong_n2_w1.log
        check pingpong_n2_w1
        echo "amoset_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 41 > $3/amoset_n2_w1_z1.log
        check amoset_n2_w1_z1
        ;;

    ###########################################################################
    ############################### SHORT TESTS ###############################
    ###########################################################################
    *"short")
        echo "get_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 0 > $3/get_n2_w16_z128_8B.log
        check get_n2_w16_z128_8B
        echo "getnbi_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 1 > $3/getnbi_n2_w16_z128_8B.log
        check getnbi_n2_w16_z128_8B
        echo "put_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 2 > $3/put_n2_w16_z128_8B.log
        check put_n2_w16_z128_8B
        echo "putnbi_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 3 > $3/putnbi_n2_w16_z128_8B.log
        check putnbi_n2_w16_z128_8B
        echo "wg_get_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 25 > $3/wg_get_n2_w1_z64_8B.log
        check wg_get_n2_w1_z64_8B
        echo "wg_getnbi_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 26 > $3/wg_getnbi_n2_w1_z64_8B.log
        check wg_getnbi_n2_w1_z64_8B
        echo "wg_put_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 27 > $3/wg_put_n2_w1_z64_8B.log
        check wg_put_n2_w1_z64_8B
        echo "wg_putnbi_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 28 > $3/wg_putnbi_n2_w1_z64_8B.log
        check wg_putnbi_n2_w1_z64_8B
        echo "wg_get_tiled_n2_w16_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 64 -s 8 -a 25 > $3/wg_get_tiled_n2_w16_z64_8B.log
        check wg_get_tiled_n2_w16_z64_8B
        echo "wg_getnbi_tiled_n2_w16_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 64 -s 8 -a 26 > $3/wg_getnbi_tiled_n2_w16_z64_8B.log
        check wg_getnbi_tiled_n2_w16_z64_8B
        echo "wg_put_tiled_n2_w16_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 64 -s 8 -a 27 > $3/wg_put_tiled_n2_w16_z64_8B.log
        check wg_put_tiled_n2_w16_z64_8B
        echo "wg_putnbi_tiled_n2_w16_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 64 -s 8 -a 28 > $3/wg_putnbi_tiled_n2_w16_z64_8B.log
        check wg_putnbi_tiled_n2_w16_z64_8B
        echo "wave_get_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 29 > $3/wave_get_n2_w1_z64_8B.log
        check wave_get_n2_w1_z64_8B
        echo "wave_getnbi_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 30 > $3/wave_getnbi_n2_w1_z64_8B.log
        check wave_getnbi_n2_w1_z64_8B
        echo "wave_put_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 31 > $3/wave_put_n2_w1_z64_8B.log
        check wave_put_n2_w1_z64_8B
        echo "wave_putnbi_n2_w1_z64_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 64 -s 8 -a 32 > $3/wave_putnbi_n2_w1_z64_8B.log
        check wave_putnbi_n2_w1_z64_8B
        echo "wave_get_tiled_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 29 > $3/wave_get_tiled_n2_w16_z128_8B.log
        check wave_get_tiled_n2_w16_z128_8B
        echo "wave_getnbi_tiled_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 30 > $3/wave_getnbi_tiled_n2_w16_z128_8B.log
        check wave_getnbi_tiled_n2_w16_z128_8B
        echo "wave_put_tiled_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 31 > $3/wave_put_tiled_n2_w16_z128_8B.log
        check wave_put_tiled_n2_w16_z128_8B
        echo "wave_putnbi_tiled_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 32 > $3/wave_putnbi_tiled_n2_w16_z128_8B.log
        check wave_putnbi_tiled_n2_w16_z128_8B
        echo "amofadd_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 5 > $3/amofadd_n2_w8_z1.log
        check amofadd_n2_w8_z1
        echo "amofinc_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 6 > $3/amofinc_n2_w8_z1.log
        check amofinc_n2_w8_z1
        echo "amofetch_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 7 > $3/amofetch_n2_w8_z1.log
        check amofetch_n2_w8_z1
        echo "amofcswap_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 8 > $3/amofcswap_n2_w8_z1.log
        check amofcswap_n2_w8_z1
        echo "amoadd_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 9 > $3/amoadd_n2_w8_z1.log
        check amoadd_n2_w8_z1
        echo "amoinc_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 10 > $3/amoinc_n2_w8_z1.log
        check amoinc_n2_w8_z1
        echo "pingpong_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 13 > $3/pingpong_n2_w1.log
        check pingpong_n2_w1
        echo "amoset_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 41 > $3/amoset_n2_w8_z1.log
        check amoset_n2_w8_z1
        echo "putsignal_n2_w1_z1"
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 50 > $3/putsignal_n2_w1_z1.log
        check putsignal_n2_w1_z1
        echo "putsignalwg_n2_w2_z32"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -s 1048576 -a 52 > $3/putsignalwg_n2_w2_z32.log
        check putsignalwg_n2_w2_z32
        echo "putsignalwave_n2_w1_z32"
        mpirun -np 2 $1 -w 1 -z 32 -s 1048576 -a 52 > $3/putsignalwave_n2_w1_z32.log
        check putsignalwave_n2_w1_z32
        echo "putsignalnbi"
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 53 > $3/putsignalnbi_n2_w1_z1.log
        check putsignalnbi_n2_w1_z1
        echo "putsignalnbiwg"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -s 1048576 -a 54 > $3/putsignalnbiwg_n2_w2_z32.log
        check putsignalnbiwg_n2_w2_z32
        echo "putsignalnbiwave"
        mpirun -np 2 $1 -w 1 -z 32 -s 1048576 -a 55 > $3/putsignalnbiwave_n2_w1_z32.log
        check putsignalnbiwave_n2_w1_z32
        echo "signalfetch"
        mpirun -np 2 $1 -w 1 -z 1 -a 56 > $3/signalfetch_n2_w1_z1.log
        check signalfetch_n2_w1_z1
        echo "signalfetchwg"
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -a 57 > $3/signalfetchwg_n2_w2_z32.log
        check signalfetchwg_n2_w2_z32
        echo "signalfetchwave"
        mpirun -np 2 $1 -w 1 -z 32 -a 57 > $3/signalfetchwave_n2_w2_z32.log
        check signalfetchwave_n2_w2_z32
        ;;

    ###########################################################################
    ############################# EXHAUSTIVE TESTS ############################
    ###########################################################################
    *"exhaustive")
        ############################### GET ###################################
        echo "get_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 0 > $3/get_n2_w1_z1_1MB.log
        check get_n2_w1_z1_1MB
        echo "get_n2_w1_z1024_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -s 512 -a 0 > $3/get_n2_w1_z1024_512B.log
        check get_n2_w1_z1024_512B
        echo "get_n2_w8_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -s 1048576 -a 0 > $3/get_n2_w8_z1_1MB.log
        check get_n2_w8_z1_1MB
        echo "get_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 0 > $3/get_n2_w16_z128_8B.log
        check get_n2_w16_z128_8B
        echo "get_n2_w32_z256_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 256 -s 512 -a 0 > $3/get_n2_w32_z256_512B.log
        check get_n2_w32_z256_512B
        echo "get_n2_w64_z1024_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=64 mpirun -np 2 $1 -w 64 -z 1024 -s 8 -a 0 > $3/get_n2_w64_z1024_8B.log
        check get_n2_w64_z1024_8B
        ############################### GETNBI ################################
        echo "getnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 1 > $3/getnbi_n2_w1_z1_1MB.log
        check getnbi_n2_w1_z1_1MB
        echo "getnbi_n2_w1_z1024_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -s 512 -a 1 > $3/getnbi_n2_w1_z1024_512B.log
        check getnbi_n2_w1_z1024_512B
        echo "getnbi_n2_w8_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -s 1048576 -a 1 > $3/getnbi_n2_w8_z1_1MB.log
        check getnbi_n2_w8_z1_1MB
        echo "getnbi_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 1 > $3/getnbi_n2_w16_z128_8B.log
        check getnbi_n2_w16_z128_8B
        echo "getnbi_n2_w32_z256_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 256 -s 512 -a 1 > $3/getnbi_n2_w32_z256_512B.log
        check getnbi_n2_w32_z256_512B
        echo "getnbi_n2_w64_z1024_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=64 mpirun -np 2 $1 -w 64 -z 1024 -s 8 -a 1 > $3/getnbi_n2_w64_z1024_8B.log
        check getnbi_n2_w64_z1024_8B
        ############################### PUT ###################################
        echo "put_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 2 > $3/put_n2_w1_z1_1MB.log
        check put_n2_w1_z1_1MB
        echo "put_n2_w1_z1024_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -s 512 -a 2 > $3/put_n2_w1_z1024_512B.log
        check put_n2_w1_z1024_512B
        echo "put_n2_w8_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -s 1048576 -a 2 > $3/put_n2_w8_z1_1MB.log
        check put_n2_w8_z1_1MB
        echo "put_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 2 > $3/put_n2_w16_z128_8B.log
        check put_n2_w16_z128_8B
        echo "put_n2_w32_z256_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 256 -s 512 -a 2 > $3/put_n2_w32_z256_512B.log
        check put_n2_w32_z256_512B
        echo "put_n2_w64_z1024_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=64 mpirun -np 2 $1 -w 64 -z 1024 -s 8 -a 2 > $3/put_n2_w64_z1024_8B.log
        check put_n2_w64_z1024_8B
        ############################### PUTNBI ################################
        echo "putnbi_n2_w1_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 3 > $3/putnbi_n2_w1_z1_1MB.log
        check putnbi_n2_w1_z1_1MB
        echo "putnbi_n2_w1_z1024_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -s 512 -a 3 > $3/putnbi_n2_w1_z1024_512B.log
        check putnbi_n2_w1_z1024_512B
        echo "putnbi_n2_w8_z1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -s 1048576 -a 3 > $3/putnbi_n2_w8_z1_1MB.log
        check putnbi_n2_w8_z1_1MB
        echo "putnbi_n2_w16_z128_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=16 mpirun -np 2 $1 -w 16 -z 128 -s 8 -a 3 > $3/putnbi_n2_w16_z128_8B.log
        check putnbi_n2_w16_z128_8B
        echo "putnbi_n2_w32_z256_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 256 -s 512 -a 3 > $3/putnbi_n2_w32_z256_512B.log
        check putnbi_n2_w32_z256_512B
        echo "putnbi_n2_w64_z1024_8B"
        ROCSHMEM_MAX_NUM_CONTEXTS=64 mpirun -np 2 $1 -w 64 -z 1024 -s 8 -a 3 > $3/putnbi_n2_w64_z1024_8B.log
        check putnbi_n2_w64_z1024_8B
        ############################## AMOFADD ###############################
        echo "amofadd_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 5 > $3/amofadd_n2_w1_z1.log
        check amofadd_n2_w1_z1
        echo "amofadd_n2_w1_z1024"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -a 5 > $3/amofadd_n2_w1_z1024.log
        check amofadd_n2_w1_z1024
        echo "amofadd_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 5 > $3/amofadd_n2_w8_z1.log
        check amofadd_n2_w8_z1
        echo "amofadd_n2_w32_z128"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 128 -a 5 > $3/amofadd_n2_w32_z128.log
        check amofadd_n2_w32_z128
        ############################## AMOFINC ###############################
        echo "amofinc_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 6 > $3/amofinc_n2_w1_z1.log
        check amofinc_n2_w1_z1
        echo "amofinc_n2_w1_z1024"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -a 6 > $3/amofinc_n2_w1_z1024.log
        check amofinc_n2_w1_z1024
        echo "amofinc_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 6 > $3/amofinc_n2_w8_z1.log
        check amofinc_n2_w8_z1
        echo "amofinc_n2_w32_z128"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 128 -a 6 > $3/amofinc_n2_w32_z128.log
        check amofinc_n2_w32_z128
        ############################ AMOFETCH ################################
        echo "amofetch_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 7 > $3/amofetch_n2_w1_z1.log
        check amofetch_n2_w1_z1
        echo "amofetch_n2_w1_z1024"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -a 7 > $3/amofetch_n2_w1_z1024.log
        check amofetch_n2_w1_z1024
        echo "amofetch_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 7 > $3/amofetch_n2_w8_z1.log
        check amofetch_n2_w8_z1
        echo "amofetch_n2_w32_z128"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 128 -a 7 > $3/amofetch_n2_w32_z128.log
        check amofetch_n2_w32_z128
        ########################### AMOFCSWAP ################################
        echo "amofcswap_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 8 > $3/amofcswap_n2_w1_z1.log
        check amofcswap_n2_w1_z1
        echo "amofcswap_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 8 > $3/amofcswap_n2_w8_z1.log
        check amofcswap_n2_w8_z1
        echo "amofcswap_n2_w32_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 1 -a 8 > $3/amofcswap_n2_w32_z1.log
        check amofcswap_n2_w32_z1
        ############################# AMOADD ################################
        echo "amoadd_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 9 > $3/amoadd_n2_w1_z1.log
        check amoadd_n2_w1_z1
        echo "amoadd_n2_w1_z1024"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -a 9 > $3/amoadd_n2_w1_z1024.log
        check amoadd_n2_w1_z1024
        echo "amoadd_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 9 > $3/amoadd_n2_w8_z1.log
        check amoadd_n2_w8_z1
        echo "amoadd_n2_w32_z128"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 128 -a 9 > $3/amoadd_n2_w32_z128.log
        check amoadd_n2_w32_z128
        ############################# AMOINC ################################
        echo "amoinc_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 10 > $3/amoinc_n2_w1_z1.log
        check amoinc_n2_w1_z1
        echo "amoinc_n2_w1_z1024"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1024 -a 10 > $3/amoinc_n2_w1_z1024.log
        check amoinc_n2_w1_z1024
        echo "amoinc_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 10 > $3/amoinc_n2_w8_z1.log
        check amoinc_n2_w8_z1
        echo "amoinc_n2_w32_z128"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 128 -a 10 > $3/amoinc_n2_w32_z128.log
        check amoinc_n2_w32_z128
        ############################## INIT #################################
        echo "init_n2"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -a 12 > $3/init_n2.log
        check init_n2
        ########################### PINGPONG ################################
        echo "pingpong_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 13 > $3/pingpong_n2_w1.log
        check pingpong_n2_w1
        echo "pingpong_n2_w8"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -a 13 > $3/pingpong_n2_w8.log
        check pingpong_n2_w8
        echo "pingpong_n2_w32"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -a 13 > $3/pingpong_n2_w32.log
        check pingpong_n2_w32
        ############################ BARRIER ################################
        echo "barrier_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 15 > $3/barrier_n2_w1.log
        check barrier_n2_w1
        ############################ SYNCALL ################################
        echo "syncall_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 16 > $3/syncall_n2_w1.log
        check syncall_n2_w1
        ############################# SYNC ##################################
        echo "sync_n2_w1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -a 17 > $3/sync_n2_w1.log
        check sync_n2_w1
        ########################### FCOLLECT ################################
        echo "fcollect_n2_w1_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -s 512 -a 19 > $3/fcollect_n2_w1_512B.log
        check fcollect_n2_w1_512B
        ########################### ALLTOALL ################################
        echo "alltoall_n2_w1_512B"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -s 512 -a 20 > $3/alltoall_n2_w1_512B.log
        check alltoall_n2_w1_512B
        ########################## TEAMGETNBI ###############################
        echo "teamgetnbi_n2_w1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -s 1048576 -a 36 > $3/teamgetnbi_n2_w1_1MB.log
        check teamgetnbi_n2_w1_1MB
        ########################## TEAMPUTNBI ###############################
        echo "teamputnbi_n2_w1_1MB"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -s 1048576 -a 38 > $3/teamputnbi_n2_w1_1MB.log
        check teamputnbi_n2_w1_1MB
        ############################ AMOSET #################################
        echo "amoset_n2_w1_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=1 mpirun -np 2 $1 -w 1 -z 1 -a 41 > $3/amoset_n2_w1_z1.log
        check amoset_n2_w1_z1
        echo "amoset_n2_w8_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=8 mpirun -np 2 $1 -w 8 -z 1 -a 41 > $3/amoset_n2_w8_z1.log
        check amoset_n2_w8_z1
        echo "amoset_n2_w32_z1"
        ROCSHMEM_MAX_NUM_CONTEXTS=32 mpirun -np 2 $1 -w 32 -z 1 -a 41 > $3/amoset_n2_w32_z1.log
        check amoset_n2_w32_z1
        ;;

    ###########################################################################
    ############################# INDIVIDUAL TESTS ############################
    ###########################################################################
    *"get")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 0
        ;;
    *"getnbi")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 1
        ;;
    *"put")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 2
        ;;
    *"putnbi")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 3
        ;;
    *"reduction")
        mpirun -np 2 $1 -w 1 -z 1 -s 32768 -a 5
        ;;
    *"amofadd")
        mpirun -np 2 $1 -w 1 -z 1 -a 5
        ;;
    *"amofinc")
        mpirun -np 2 $1 -w 1 -z 1 -a 6
        ;;
    *"amofetch")
        mpirun -np 2 $1 -w 1 -z 1 -a 7
        ;;
    *"amofcswap")
        mpirun -np 2 $1 -w 1 -z 1 -a 8
        ;;
    *"amoadd")
        mpirun -np 2 $1 -w 1 -z 1 -a 9
        ;;
    *"amoinc")
        mpirun -np 2 $1 -w 1 -z 1 -a 10
        ;;
    *"init")
        mpirun -np 2 $1 -a 12
        ;;
    *"pingpong")
        mpirun -np 2 $1 -w 1 -z 1 -a 13
        ;;
    *"barrier")
        mpirun -np 2 $1 -w 1 -z 1 -a 15
        ;;
    *"syncall")
        mpirun -np 2 $1 -w 1 -z 1 -a 16
        ;;
    *"sync")
        mpirun -np 2 $1 -w 1 -z 1 -s 8 -a 17
        ;;
    *"fcollect")
        mpirun -np 2 $1 -w 1 -z 1 -s 32768 -a 19
        ;;
    *"alltoall")
        mpirun -np 2 $1 -w 1 -z 1 -s 32768 -a 20
        ;;
    *"team_broadcast")
        mpirun -np 2 $1 -w 1 -z 1 -s 32768 -a 33
        ;;
    *"team_reduction")
        mpirun -np 2 $1 -w 1 -z 1 -s 32768 -a 34
        ;;
    *"team_get")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 35
        ;;
    *"team_getnbi")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 36
        ;;
    *"team_put")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 37
        ;;
    *"team_putnbi")
        mpirun -np 2 $1 -w 1 -z 1 -s 1048576 -a 38
        ;;
    *"ctx_infra")
        mpirun -np 2 $1 -w 1 -z 1 -a 39
        ;;
    *"amoset")
        mpirun -np 2 $1 -w 1 -z 1 -a 41
        ;;
    *"amoswap")
        mpirun -np 2 $1 -w 1 -z 1 -a 42
        ;;
    *"amofetchand")
        mpirun -np 2 $1 -w 1 -z 1 -a 43
        ;;
    *"amofetchor")
        mpirun -np 2 $1 -w 1 -z 1 -a 44
        ;;
    *"amofetchxor")
        mpirun -np 2 $1 -w 1 -z 1 -a 45
        ;;
    *"amoand")
        mpirun -np 2 $1 -w 1 -z 1 -a 46
        ;;
    *"amoor")
        mpirun -np 2 $1 -w 1 -z 1 -a 47
        ;;
    *"amoxor")
        mpirun -np 2 $1 -w 1 -z 1 -a 48
        ;;
    *"putsignal")
        mpirun -np 2 $1 -w 1 -z 1 -a 50
        ;;
    *"putsignalwg")
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -a 52
        ;;
    *"putsignalwave")
        mpirun -np 2 $1 -w 1 -z 32 -a 52
        ;;
    *"putsignalnbi")
        mpirun -np 2 $1 -w 1 -z 1 -a 53
        ;;
    *"putsignalnbiwg")
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -a 54
        ;;
    *"putsignalnbiwave")
        mpirun -np 2 $1 -w 1 -z 32 -a 55
        ;;
    *"signalfetch")
        mpirun -np 2 $1 -w 1 -z 1 -a 56
        ;;
    *"signalfetchwg")
        ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 $1 -w 2 -z 32 -a 57
        ;;
    *"signalfetchwave")
        mpirun -np 2 $1 -w 1 -z 32 -a 57
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $(($driver_return_status || $?))
