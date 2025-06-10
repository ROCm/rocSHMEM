#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

#include <rocm-core/rocm_version.h>
#include <rocshmem/rocshmem.hpp>

#define NAME_COLUMN_WIDTH (24)
#define INFO_COLUMN_WIDTH (51)

#define PRINT_ENTRY(NAME, INFO)   \
  printf("# %-*s: %-*s#\n", NAME_COLUMN_WIDTH, NAME, INFO_COLUMN_WIDTH, INFO)

const std::string config_file_path = std::string(ROCSHMEM_INSTALL_PREFIX)
                                   + std::string("/include/rocshmem/rocshmem_config.h");

bool is_config_file_valid() {
  std::ifstream file;

  if (file.is_open()) {
    fprintf(stderr, "Could not open config file: %s\n", config_file_path.c_str());
    return false;
  }

  file.open(config_file_path);
  if (!file.is_open()) {
    fprintf(stderr, "Error opening config file: %s\n", config_file_path.c_str());
    return false;
  }

  file.close();

  return true;
}

void parse_config_file() {
  std::string line;
  std::ifstream file;

  const std::string define     = "#define ";
  const std::string undef_pre  = "/* #undef ";
  const std::string undef_post = " */";

  printf("#------------------------------------------------------------------------------#\n");
  printf("#                              Build Configuration                             #\n");
  printf("#------------------------------------------------------------------------------#\n");

  file.open(config_file_path);

  while (std::getline(file, line)) {
    if (line.find(undef_pre) != std::string::npos) {
      line.replace(line.find(undef_pre), undef_pre.length(), "");
      line.replace(line.find(undef_post), undef_post.length(), "");
      PRINT_ENTRY(line.c_str(), "OFF");
    }

    if (line.find(define) != std::string::npos) {
      line.replace(line.find(define), define.length(), "");
      PRINT_ENTRY(line.c_str(), "ON");
    }
  }

  file.close();
}

void print_mpi_info() {
#ifdef OMPI_MAJOR_VERSION
  char mpi_version[8];
  snprintf(mpi_version, sizeof(mpi_version), "%d.%d.%d",
           OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
  PRINT_ENTRY("Open MPI", mpi_version);
#else
  PRINT_ENTRY("MPI ", "Unsupported MPI Library");
#endif
}

void print_rocm_info() {
  char rocm_version[32];
  snprintf(rocm_version, sizeof(rocm_version), "%d.%d.%d",
           ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, ROCM_VERSION_PATCH);
  PRINT_ENTRY("ROCm", rocm_version);
}

int main (int argc, char **argv) {

  printf("################################################################################\n");
  printf("#                                rocSHMEM Info                                 #\n");
  printf("################################################################################\n");

  PRINT_ENTRY("Version", rocshmem::VERSION);
  PRINT_ENTRY("Git Hash", ROCSHMEM_GIT_HASH);
  PRINT_ENTRY("Install Prefix", ROCSHMEM_INSTALL_PREFIX);

  print_rocm_info();
  print_mpi_info();

  if (is_config_file_valid()) {
    parse_config_file();
  }

  printf("################################################################################\n");
  return 0;
}
