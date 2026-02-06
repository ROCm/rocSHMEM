###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

#!/bin/bash
if true || tty -s; then
  PRETTY_FAILED="\033[1;31mFAILED\033[0m"
  PRETTY_PASSED="\033[1;32mPASSED\033[0m"
else
  PRETTY_FAILED="FAILED"
  PRETTY_PASSED="PASSED"
fi

# This names/values should match the TestType enum in rocSHMEM/tests/functional_tests/tester.hpp
declare -A TEST_NUMBERS=(
  ["allreducepush"]="0"
  ["fusedgemmallreducepush"]="1"
)

declare -A TEST_KERNELS=(
  ["pingpong"]="PingPong"
  ["allreducepush"]="Allreduce"
  ["fusedgemmallreducepush"]="FusedGemmAllreduce"
)

ExecTest() {
  TEST_NAME=$1
  NUM_RANKS=$2
  NUM_WG=$3
  NUM_THREADS=$4
  MAX_MSG_SIZE=$5
  TIMEOUT=$((5 * 60)) # Timeout in seconds

  pkill rocshmem_relax

  if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(amd-smi list | grep GPU | wc -l)}
  elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(rocm-smi --showserial | grep GPU | wc -l)}
  fi
  NUM_GPUS=${NUM_GPUS:-0}
  NUM_GPUS=$(($NUM_GPUS > 0? $NUM_GPUS: 8))

  # Multi-node: if num_ranks > 8, use GDA backend and multi-node hostfile.
  if [ $NUM_RANKS -gt 8 ]; then
    HOSTFILE=/root/hostfile.txt
    ENABLE_MULTINODE=1
  else
    ENABLE_MULTINODE=0
  fi

  TEST_NUM=${TEST_NUMBERS[$TEST_NAME]}
  TEST_KERN=${TEST_KERNELS[$TEST_NAME]}

  if [[ "" == "$TEST_NUM" ]]
  then
    echo "Test $TEST_NAME does not exist" >&2
    DRIVER_RETURN_STATUS=1
    return
  fi

  ELEM_PER_THREAD=1
  if [[ "$TEST_NAME" == "fusedgemmallreducepush" ]]
  then
    # Override num_wgs with = gemm_m * gemm_n / (num_ranks * wg_size) / elem_per_thread
    # At this point: $2=NUM_RANKS, $4=NUM_THREADS(wg_size), $5=GEMM_N, $6=GEMM_M.
    NUM_WG=$(( $6 * $5 / ($2 * $4 * $ELEM_PER_THREAD) ))
    echo "## set NUM_WG to $NUM_WG"
  else
    BYTE_PER_THREAD=$(( $5 / ($4) ))
    TOTAL_BYTES=$(( $5 * $2 * $NUM_WG ))
    echo "## Each thread copies $BYTE_PER_THREAD bytes, total $TOTAL_BYTES bytes"
  fi
  ROCSHMEM_MAX_NUM_CONTEXTS=$(( 2 * $NUM_WG))

  # MPI Parameters
  LAUNCHER=mpirun
  OPTIONS=" -n $NUM_RANKS -mca pml ucx -mca osc ucx "
  OPTIONS+=" --allow-run-as-root "
  OPTIONS+=" -x ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS"
  OPTIONS+=" -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
  if [ $ENABLE_MULTINODE -eq 1 ]; then
    OPTIONS+=" -x ROCSHMEM_BACKEND=gda"
    OPTIONS+=" --map-by node:PE=14 --rank-by slot --display-map  --prtemca plm ssh --prtemca plm_ssh_args \"-l root -p 2222 \" --prtemca plm_ssh_no_tre_spawn 1 --prtemca oob_tcp_if_include ens14np0 -x PMIX_MCA_ptl_tcp_if_include=ens14np0 -x PATH -x LD_LIBRARY_PATH -mca ras_base_verbose 100 --mca rmaps_base_verbose 100"
  else
    OPTIONS+=" --map-by numa"
  fi
  OPTIONS+=" --timeout $TIMEOUT"

  if [[ "" != "$ROCSHMEM_TEST_USE_DEFAULT_STREAM" ]]
  then
    OPTIONS+=" -x ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"
  fi

  if [[ "" != "$HOSTFILE" ]]
  then
    OPTIONS+=" --hostfile $HOSTFILE"
  fi

  # Construct Test Command
  TEST_LOG_NAME="$TEST_NAME"_n"$NUM_RANKS"_w"$NUM_WG"_z"$NUM_THREADS"

  # if using running fusedgemm, name is based on m and n values
  GEMM_M=1
  GEMM_N=1
  if [[ "$TEST_NAME" == "fusedgemm"* ]]
  then
    GEMM_N=$5
    GEMM_M=$6
    TEST_LOG_NAME+="_m${GEMM_M}_n${GEMM_N}"
  elif [[ "" != "$MAX_MSG_SIZE" ]]
  then
    TEST_LOG_NAME+=_"$MAX_MSG_SIZE"B
  fi

  TRACE_DEST="../rocprof_out/"

  SKIP=10
  LOOPS=200
  if [ $ENABLE_MULTINODE -eq 1 ]; then
    GDA_CMD="ROCSHMEM_BACKEND=gda $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS -n $LOOPS -nskip $SKIP "
    GDA_RLX_CMD="ROCSHMEM_BACKEND=gda ROCSHMEM_TARGETED_ORDER=1 $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS  -n $LOOPS -nskip $SKIP "
  else
    GDA_CMD="ROCSHMEM_BACKEND=gda ROCSHMEM_DISABLE_MIXED_IPC=1 $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS -n $LOOPS -nskip $SKIP "
    GDA_RLX_CMD="ROCSHMEM_BACKEND=gda ROCSHMEM_DISABLE_MIXED_IPC=1 ROCSHMEM_TARGETED_ORDER=1 $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS  -n $LOOPS -nskip $SKIP "
  fi
  IPC_CMD="ROCSHMEM_BACKEND=ipc $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS  -n $LOOPS -nskip $SKIP "
  IPC_RLX_CMD="ROCSHMEM_BACKEND=ipc ROCSHMEM_TARGETED_ORDER=1 $LAUNCHER $OPTIONS $APP -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS  -n $LOOPS -nskip $SKIP "

  if [[ "$TEST_NAME" == "fusedgemm"* ]]
  then
    GDA_CMD+=" -gm $GEMM_M -gn $GEMM_N "
    GDA_RLX_CMD+=" -gm $GEMM_M -gn $GEMM_N "
    IPC_CMD+=" -gm $GEMM_M -gn $GEMM_N "
    IPC_RLX_CMD+=" -gm $GEMM_M -gn $GEMM_N "
  fi

  if [[ "" != "$MAX_MSG_SIZE" ]]
  then
    GDA_CMD+=" -s $MAX_MSG_SIZE"
    GDA_RLX_CMD+=" -s $MAX_MSG_SIZE"
    IPC_CMD+=" -s $MAX_MSG_SIZE"
    IPC_RLX_CMD+=" -s $MAX_MSG_SIZE"
  fi

  GDA_CMD+=" > $LOG_DIR/${TEST_LOG_NAME}_gda.log 2>&1"
  GDA_RLX_CMD+=" > $LOG_DIR/${TEST_LOG_NAME}_gda_rlx.log 2>&1"
  IPC_CMD+=" > $LOG_DIR/${TEST_LOG_NAME}_ipc.log 2>&1"
  IPC_RLX_CMD+=" > $LOG_DIR/${TEST_LOG_NAME}_ipc_rlx.log 2>&1"
  ALL_ERRORS=0
  # Run Test
  if [ $NUM_GPUS -ge $NUM_RANKS ] || [[ "" != "$HOSTFILE" ]]; then
    if [ $ENABLE_MULTINODE -eq 1 ]; then
      # GDA unrelaxed
      echo ${TEST_LOG_NAME}_gda
      echo "# $GDA_CMD" >"$LOG_DIR/${TEST_LOG_NAME}_gda.log"
      eval $GDA_CMD
      ALL_ERRORS+=$?
      # GDA relaxed
      echo ${TEST_LOG_NAME}_gda_rlx
      echo "# $GDA_RLX_CMD" >"$LOG_DIR/${TEST_LOG_NAME}_gda_rlx.log"
      eval $GDA_RLX_CMD
      ALL_ERRORS+=$?
    else
      # IPC unrelaxed
      echo ${TEST_LOG_NAME}_ipc
      echo "# $IPC_CMD" >"$LOG_DIR/${TEST_LOG_NAME}_ipc.log"
      eval $IPC_CMD
      ALL_ERRORS+=$?
      # IPC relaxed
      echo ${TEST_LOG_NAME}_ipc_rlx
      echo "# $IPC_RLX_CMD" >"$LOG_DIR/${TEST_LOG_NAME}_ipc_rlx.log"
      eval $IPC_RLX_CMD
      ALL_ERRORS+=$?
    fi
  else
    echo "Skipping test $TEST_LOG_NAME ($NUM_RANKS greater than $NUM_GPUS)"
  fi

  # Validate Test
  if [ $ALL_ERRORS -ne 0 ]
  then
    echo -e "$PRETTY_FAILED: $TEST_LOG_NAME" >&2
    if [ $ENABLE_MULTINODE -eq 1 ]; then
      cat "$LOG_DIR/${TEST_LOG_NAME}_gda.log"
      cat "$LOG_DIR/${TEST_LOG_NAME}_gda_rlx.log"
    else
      cat "$LOG_DIR/${TEST_LOG_NAME}_ipc.log"
      cat "$LOG_DIR/${TEST_LOG_NAME}_ipc_rlx.log"
    fi
    DRIVER_RETURN_STATUS=1
    FAILED_LIST="$FAILED_LIST $TEST_LOG_NAME"
  fi

  unset ROCSHMEM_MAX_NUM_CONTEXTS
}

TestRelax() {
  ################################################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size/Gemm_N [| Gemm_M] #
  ################################################################################################
  ExecTest  "allreducepush"    8       128            512        128
  ExecTest  "allreducepush"    8       128            512        256
  ExecTest  "allreducepush"    8       128            512        512 
  ExecTest  "allreducepush"    8       128            512        1024
  ExecTest  "allreducepush"    8       128            512        2048
  ExecTest  "allreducepush"    8       128            512        4096
  ExecTest  "allreducepush"    8       128            512        8192
  ExecTest  "allreducepush"    8       128            512        16384
  ExecTest  "allreducepush"    8       128            512        32768
  ExecTest  "allreducepush"    8       128            512        65536
  ExecTest  "allreducepush"    8       128            512        131072
  ExecTest  "allreducepush"    8       128            512        262144

  ExecTest  "fusedgemmallreducepush"    8      64            128        4096             64
  ExecTest  "fusedgemmallreducepush"    8      128            128        8192             64
  ExecTest  "fusedgemmallreducepush"    8      256            128        16384             64
  ExecTest  "fusedgemmallreducepush"    8      64            128        4096             128
  ExecTest  "fusedgemmallreducepush"    8      128            128        8192             128
  ExecTest  "fusedgemmallreducepush"    8      256            128        16384             128
  ExecTest  "fusedgemmallreducepush"    8      64            128        4096             256
  ExecTest  "fusedgemmallreducepush"    8      128            128        8192             256
  ExecTest  "fusedgemmallreducepush"    8      256            128        16384             256
}

ValidateInput() {
  INPUT_COUNT=$1
  if [ $INPUT_COUNT -lt 3 ] ; then
    echo "This script must be run with at least 3 arguments."
    echo 'Usage: ${0} argument1 argument2 argument3 [argument4]'
    echo "  argument1 : path to the tester driver"
    echo "  argument2 : test type to run, e.g put"
    echo "  argument3 : directory to put the output logs"
    echo "  argument4 : path to hostfile"
    exit 1
  fi
}

ValidateLogDir() {
  if [ ! -d $1 ]; then
    echo "LOG_DIR=$1 does not exist"
    mkdir -p $1
    echo "Created $1"
  fi
}

APP=$1
TEST=$2
LOG_DIR=$3
HOSTFILE=$4

DRIVER_RETURN_STATUS=0

ValidateInput $#
ValidateLogDir $LOG_DIR

case $TEST in
  *"all")
    TestRelax
    ;;
  *)
    ##############################################################################
    #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
    ##############################################################################
    ExecTest  $TEST              2       1            1         8
    ;;
esac

EXIT_STATUS=$(($DRIVER_RETURN_STATUS || $?))
if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "TESTS PASSED"
else
  echo -e "TESTS FAILED: $FAILED_LIST"
fi
exit $EXIT_STATUS
