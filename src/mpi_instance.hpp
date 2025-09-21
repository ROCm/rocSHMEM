/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_MPI_INSTANCE_HPP_
#define LIBRARY_SRC_MPI_INSTANCE_HPP_

#include <mpi.h>

#include <memory>

/**
 * @file mpi_instance.hpp
 *
 * @brief Contains MPI library initialization code
 */

namespace rocshmem {

//Open MPI related declarations if we do not find MPI header files
#define MPI_Comm    void*
#define MPI_Win     void*
#define MPI_Group   void*
#define MPI_Op      void*
#define MPI_Datatye void*
#define MPI_Request void*
#define MPI_Aint    uint64_t

#define MPI_UNDEFINED -32766
#define MPI_THREAD_MULTIPLE 3
#define MPI_SUCCESS 0
#define MPI_IN_PLACE (void*)1

#define MPI_Aint_diff(addr1, addr2) ((MPI_Aint) ((char *) (addr1) - (char *) (addr2)))

struct mpilib_funcs_t {
  int (*Init_thread)(int *argc, char ***argv, int required, int *provided);
  int (*Initialized)(int *flag);
  int (*Finalize)(void);
  int (*Finalized)(int *flag);
  int (*Comm_rank)(MPI_Comm comm, int *rank);
  int (*Comm_size)(MPI_Comm comm, int *size);
  int (*Abort)(MPI_Comm comm, int errorcode);
  int (*Get_address)(const void *location, MPI_Aint *address);
  int (*Type_size)(MPI_Datatype type, int *size);
  int (*Iprobe)(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
  int (*Testsome)(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[],
                  MPI_Status array_of_statuses[]);
  int (*Comm_split)(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
  int (*Comm_split_type)(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm);
  int (*Comm_group)(MPI_Comm comm, MPI_Group *group);
  int (*Comm_create_group)(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm);
  int (*Comm_dup)(MPI_Comm comm, MPI_Comm *newcomm);
  int (*Comm_free)(MPI_Comm *comm);
  int (*Group_free)(MPI_Group *group);
  int (*Group_translate_ranks)(MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[]);
  int (*Group_incl)(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);
  int (*Allgather)(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
  int (*Allreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                   MPI_Op op, MPI_Comm comm);
  int (*Alltoall)(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm);
  int (*Bcast)(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
  int (*Barrier)(MPI_Comm comm);
  int (*Iallreduce)(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    MPI_Op op, MPI_Comm comm, MPI_Request *request);
  int (*Ibarrier)(MPI_Comm comm, MPI_Request *request);
  int (*Win_create)(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
  int (*Win_free)(MPI_Win *win);
  int (*Win_flush)(MPI_Win win);
  int (*Win_flush_all)(MPI_Win win);
  int (*Win_flush_local)(int rank, MPI_Win win);
  int (*Win_lock)(int lock_type, int rank, int mpi_assert, MPI_Win win);
  int (*Win_lock_all)(int mpi_assert, MPI_Win win);
  int (*Win_sync)(MPI_Win win);
  int (*Win_unlock)(int rank, MPI_Win win);
  int (*Win_unlock_all)(MPI_Win win);
  int (*Get)(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,
             MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win);
  int (*Rget)(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_count, MPI_Datatype target_datatype,  MPI_Win win, MPI_Request *request);
  int (*Put)(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
             int target_count, MPI_Datatype target_datatype, MPI_Win win);
  int (*Rput)(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
              int target_cout, MPI_Datatype target_datatype, MPI_Win win, MPI_Request *request);
  int (*Compare_and_swap)(const void *origin_addr, const void *compare_addr, void *result_addr, MPI_Datatype datatype, int target_rank,
                          MPI_Aint target_disp, MPI_Win win);
  int (*Fetch_and_op)(const void *origin_addr, void *result_addr, MPI_Datatype datatype,
                      int target_rank, MPI_Aint target_disp, MPI_Op op, MPI_Win win);
  /* Open MPI specific symbols */
  void *ompi_mpi_comm_world;
  void *ompi_mpi_comm_null;
  void *ompi_request_null;
  void *ompi_mpi_datatype_null;

  void *ompi_mpi_op_max;
  void *ompi_mpi_op_min;
  void *ompi_mpi_op_sum;
  void *ompi_mpi_op_prod;
  void *ompi_mpi_op_band;
  void *ompi_mpi_op_bor;
  void *ompi_mpi_op_bxor;
  void *ompi_mpi_op_replace;

  void *ompi_mpi_char;
  void *ompi_mpi_unsigned_char;
  void *ompi_mpi_signed_char;
  void *ompi_mpi_short;
  void *ompi_mpi_int;
  void *ompi_mpi_long;
  void *ompi_mpi_unsigned_long;
  void *ompi_mpi_long_long_int;
  void *ompi_mpi_float;
  void *ompi_mpi_double;
  void *ompi_mpi_long_double;
}
struct mpilib_funcs_t mpilib_ftable_;
#define OMPI_PREDEFINED_GLOBAL(type, global) (static_cast<type> (static_cast<void *> (global)))

#define MPI_COMM_WORLD OMPI_PREDEFINED_GLOBAL(MPI_Comm, mpilib_ftable_.ompi_mpi_comm_world)
#define MPI_COMM_NULL OMPI_PREDEFINED_GLOBAL(MPI_Comm, mpilib_ftable_.ompi_mpi_comm_null)
#define MPI_REQUEST_NULL OMPI_PREDEFINED_GLOBAL(MPI_Request, mpilib_ftable_.ompi_request_null)
#define MPI_WIN_NULL OMPI_PREDEFINED_GLOBAL(MPI_Win, mpilib_ftable_.ompi_mpi_win_null)

#define MPI_MAX OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_max)
#define MPI_MIN OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_min)
#define MPI_SUM OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_sum)
#define MPI_PROD OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_prod)
#define MPI_BAND OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_band)
#define MPI_BOR OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_bor)
#define MPI_BXOR OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_bxor)
#define MPI_REPLACE OMPI_PREDEFINED_GLOBAL(MPI_Op, mpilib_ftable_.ompi_mpi_op_replace)

#define MPI_DATATYPE_NULL OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_datatype_null)
#define MPI_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_char)
#define MPI_UNSIGNED_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_unsigned_char)
#define MPI_SIGNED_CHAR OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_signed_char)
#define MPI_SHORT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_short)
#define MPI_INT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_int)
#define MPI_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_long)
#define MPI_UNSIGNED_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_unsigned_long)
#define MPI_LONG_LONG OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_long_long_int)
#define MPI_FLOAT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_float)
#define MPI_DOUBLE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_double)
#define MPI_LONG_DOUBLE OMPI_PREDEFINED_GLOBAL(MPI_Datatype, mpilib_ftable_.ompi_mpi_long_double)

class MPIInstance {
  public:
    /**
     * @brief Primary constructor
     */
    MPIInstance(MPI_Comm comm);

    /**
     * @brief Destructor
     */
    ~MPIInstance();

    /**
     * @brief Accessor for my COMM_WORLD rank identifier
     *
     * @return My COMM_WORLD rank identifier
     */
    int get_rank();

    /**
     * @brief Accessor for number or processes in COMM_WORLD
     *
     * @return Number of processes in COMM_WORLD
     */
    int get_nprocs();

  private:
    /**
     * @brief My MPI rank identifier
     */
    int my_rank_{-1};

    /**
     * @brief Number of MPI processes
     */
    int nprocs_{-1};

    /**
     * @brief Was MPI initialized in this class
     */
    int init_in_this_class{0};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MPI_INSTANCE_HPP_
