HIPCC=hipcc
BUILD=./build
SRC=./src
RESULTS=./results

#rocshmem_DIR=${HOME}/rocshmem
#MPI_HOME=${HOME}/mpich/install
NCCL_HOME=${HOME}/rccl/build

MPI_FLAGS=-lmpi -lhsa-runtime64 -lrt -L${MPI_HOME}/lib -fgpu-rdc 
SHMEM_FLAGS=${MPI_FLAGS} -lmlx5 -libverbs 
RCCL_FLAGS=${MPI_FLAGS} -Wl,-rpath,$(NCCL_HOME) -L${NCCL_HOME} -lrccl

.SILENT: run_scan extract_scan run_sort run_sort_shmem run_sort_rccl extract_sort

all: ${BUILD}/sort_shmem ${BUILD}/sort_rccl ${BUILD}/sort_mpi

${BUILD}/sort_shmem: ${BUILD}/sort_shmem.o ${rocshmem_DIR}/lib/librocshmem.a
	${HIPCC} $^ ${SHMEM_FLAGS} -o $@

${BUILD}/sort_shmem.o: ${SRC}/sort_shmem.cu
	${HIPCC} $^ -I${rocshmem_DIR}/include -I${MPI_HOME}/include -fgpu-rdc -o $@ -c

${BUILD}/sort_rccl: ${BUILD}/sort_rccl.o
	${HIPCC} $^ ${RCCL_FLAGS} -o $@

${BUILD}/sort_rccl.o: ${SRC}/sort_rccl.cu
	${HIPCC} $^ -I$(NCCL_HOME)/include/rccl -I${MPI_HOME}/include -fgpu-rdc -o $@ -c

${BUILD}/sort_mpi: ${BUILD}/sort_mpi.o
	${HIPCC} $^ ${MPI_FLAGS} -o $@

${BUILD}/sort_mpi.o: ${SRC}/sort_mpi.cu
	${HIPCC} $^ -I${MPI_HOME}/include -fgpu-rdc -o $@ -c

RO_FLAGS=ROCSHMEM_RO=1 RO_NET_CPU_QUEUE=1
ITERS?=0 1 2 3 4 5 6 7 8 9
TIMEOUT=1m
HOSTS=sv-pdp-0,sv-pdp-1,sv-pdp-2,sv-pdp-3
SCAN_SIZE=1024
PES=2 4 8 12 16
PES_RCCL=2 4 8

TYPE ?= Naive
LABEL ?= naive
PARAM ?= 0
NUM_PES ?= 2

run_sort_shmem: ${BUILD}/sort_shmem
	printf "${TYPE} ";\
	echo "" > ${RESULTS}/sort_${LABEL}_${NUM_PES}.out; \
	for j in ${ITERS}; do \
		${RO_FLAGS} timeout ${TIMEOUT} mpirun -np ${NUM_PES} -hosts ${HOSTS} ${BUILD}/sort_shmem ${PARAM} >> ${RESULTS}/sort_${LABEL}_${NUM_PES}.out;\
	done;

run_sort_rccl: ${BUILD}/sort_rccl
	printf "RCCL "; \
	echo "" > ${RESULTS}/sort_rccl_${NUM_PES}.out; \
	for j in ${ITERS}; do \
		timeout ${TIMEOUT} mpirun -np ${NUM_PES} -hosts ${HOSTS} ${BUILD}/sort_rccl >> ${RESULTS}/sort_rccl_${NUM_PES}.out;\
	done;

run_sort_mpi: ${BUILD}/sort_rccl
	printf "MPI2 "; \
	echo "" > ${RESULTS}/sort_mpi2_${NUM_PES}.out; \
	for j in ${ITERS}; do \
		timeout ${TIMEOUT} mpirun -np ${NUM_PES} -hosts ${HOSTS} ${BUILD}/sort_mpi >> ${RESULTS}/sort_mpi2_${NUM_PES}.out;\
	done;

run_sort: ${BUILD}/sort_shmem ${BUILD}/sort_rccl
	for i in ${PES}; do \
		printf "%d " $$i; \
		$(MAKE) --no-print-directory run_sort_shmem TYPE=NAIVE LABEL=naive PARAM=0 NUM_PES=$${i}; \
		$(MAKE) --no-print-directory run_sort_shmem TYPE=MPI LABEL=mpi PARAM=1 NUM_PES=$${i}; \
		$(MAKE) --no-print-directory run_sort_shmem TYPE=GCEN LABEL=gcen PARAM=2 NUM_PES=$${i}; \
		$(MAKE) --no-print-directory run_sort_shmem TYPE=GCEN2 LABEL=gcen2 PARAM=3 NUM_PES=$${i}; \
		$(MAKE) --no-print-directory run_sort_mpi NUM_PES=$${i}; \
		printf "\n";\
	done
	for i in ${PES_RCCL}; do \
		$(MAKE) --no-print-directory run_sort_rccl NUM_PES=$${i}; \
		printf "%d " $$i; \
	done

	$(MAKE) extract_sort


extract_sort:
	printf "Sort latency\n"
	printf "PROCS\tType\tRuns"
	for i in ${PES}; do \
		for type in mpi mpi2 rccl naive gcen gcen2; do\
			printf "\n%d\t$${type}\t" $$i; \
			file=${RESULTS}/sort_$${type}_$${i}.out;\
			latency=$$(grep -E "Avg time" $${file}); \
			grep -E "Avg time" $${file} | while read -r j; do\
				val=$$(echo $$j | grep -oE -m1 "[0-9]+\.[0-9]+");\
				printf "%s\t" $${val};\
			done; \
		done;\
	done
	printf "\n"

clean: 
	rm build/*;
