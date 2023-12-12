#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
#include <time.h>
int min(int a, int b) {

    return a < b ? a : b;
}
int solve(double **matrix, int n, int i, int size, int rank, MPI_Comm comm)
{
    if (rank == 0) {
        printf("\n"); fflush(stdout);
    }
    double lead[n * 2];
    if (i == 1 && rank == 1) {
        raise(SIGKILL);
    }
    int block = n / size, start_row = block * rank, end_row = block * (rank + 1);
    if (rank == size - 1)
        end_row = n;
	struct { 
	    double value; 
	    int   index; 
	} in, out; 
	in.value = -1; 
	in.index = 0; 
	for (int k=0; k < (end_row - start_row); k++) {
	    if (k + rank * block >= i) {
	    	if (in.value < fabs(matrix[k][i])) { 
				in.value = fabs(matrix[k][i]); 
				in.index = k; 
	    	} 
	    }
	}

    in.index = rank * block + in.index; 
    int err = MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
	if (err != MPI_SUCCESS) {
	    return -1;
	}
    if (out.index != i) {
        if (rank == min(i / block, size - 1) && rank == min(out.index / block, size - 1)) {
            for (int j = 0; j < 2 * n; j++) {
                double tmp = matrix[i - rank * block][j];
                matrix[i - rank*block][j] = matrix[out.index - rank*block][j];
                matrix[out.index - rank*block][j] = tmp;
            }
        } else if (rank == min(i / block, size - 1)) {
            for (int j = 0; j < 2 * n; j++) {
                MPI_Status status;
                MPI_Request request;
                double tmp = matrix[i - rank*block][j];
                MPI_Recv(&matrix[i - rank*block][j], 1, MPI_DOUBLE,  min(out.index / block, size - 1), 0, comm, &status);
            MPI_Isend(&tmp, 1, MPI_DOUBLE, min(out.index / block, size - 1), 0, comm, &request);
            }
        } else if (rank == min(out.index / block, size - 1)) {
            for (int j = 0; j < 2 * n; j++) {
                MPI_Status status;
                MPI_Request request;
                double tmp = matrix[out.index - rank*block][j];
                MPI_Isend(&tmp, 1, MPI_DOUBLE, min(i / block, size - 1), 0, comm, &request);
                MPI_Recv(&matrix[out.index - rank*block][j], 1, MPI_DOUBLE,  min(i / block, size - 1), 0, comm, &status);
            }
        }
    }
    MPI_Barrier(comm);
    if (rank == min(i / block, size - 1)) {
        for (int j = 0; j < n * 2; j++) {
            lead[j] = matrix[i - rank*block][j];
        }
    }
    MPI_Barrier(comm);
    err = MPI_Bcast(lead, n * 2, MPI_DOUBLE, min(i / block, size - 1), comm);
    if (err != MPI_SUCCESS) {
        return -1;
    }
    for (int j = 0; j < end_row - start_row; j++) {
        if (j + start_row == i) {
            double d = matrix[j][i];
            for (int k = 0; k < n * 2; k++)
                matrix[j][k] /= d;
            continue;
        }
        double d = matrix[j][i] / lead[i];
        for (int k = 0; k < n * 2; k++) {
            matrix[j][k] -= d * lead[k];
        }
    }
    return 0;
}
int malloc2dfloat(double ***array, int n, int m) {

    double *p = (double *)malloc(n*m*sizeof(double));
    if (!p) return -1;

    (*array) = (double **)malloc(n*sizeof(double*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    for (int i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}
void print_matrix(double **matrix, int n, int m)
{
  for (int i = 0; i < n; i ++) {
      for (int j = 0; j < m; j ++) {
          printf("%lf ", matrix[i][j]); fflush(stdout);
      }
      printf("\n"); fflush(stdout);
  }
}
int main(int argc, char **argv) {

    int rank, size;
    double start, end;
    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    srand(time(NULL));
    int block = n / size,
    start_row = block * rank,
    end_row = block * (rank + 1);
    if (rank == size - 1)
        end_row = n;
    double **matrix;
    malloc2dfloat(&matrix, end_row - start_row, 2*n);
    for (int i = 0; i < end_row - start_row; ++i) {
        for (int j = 0; j < 2 * n; ++j) {
            matrix[i][j] = (j < n) ? (rand() + rank) % 5 : ((j - n == i + start_row) ? 1 : 0);
        }
    }
    for (int i = 0; i < size; i ++) {
        if (rank == i) {
            print_matrix(matrix, end_row - start_row, 2*n);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_File handle;
    MPI_Status status;
    MPI_File_open(comm, "checkpoint.txt", MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &handle);
    MPI_File_write_at(handle,  start_row*2*n*sizeof(matrix[0][0]), matrix[0], (end_row - start_row) *  2 * n, MPI_DOUBLE, &status);
    MPI_File_close(&handle);     
    MPI_Barrier(comm);
    int i = 0;
    while (i < n) {
        int finished;
        int result = solve(matrix, n, i, size, rank, comm);
        MPI_Barrier(comm);
        MPI_Bcast( &result, 1, MPI_INT, 0, comm);
        if (result == 0) {
            if (rank == 0)
                printf("\nsaving checkpoint on iter %d, %d proc alive:\n", i, size); fflush(stdout);
			int block = n / size,
			start_row = block * rank,
			end_row = block * (rank + 1);
			if (rank == size - 1) {
				end_row = n;
			}
            MPI_Barrier(comm);
            MPI_File handle;
            MPI_Status status;
			MPI_File_open(comm, "checkpoint.txt", MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &handle);
			MPI_File_write_at(handle, start_row*2*n*sizeof(matrix[0][0]), matrix[0], (end_row - start_row) * 2*n, MPI_DOUBLE, &status);
			MPI_Barrier(comm);
			MPI_File_close(&handle);
            i++;
         } else {
            MPIX_Comm_shrink(comm, &comm);
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            if (rank == 0) 
                printf("\nloading checkpoint on iter %d, %d proc alive:\n", i, size); fflush(stdout);
			free(matrix[0]);
			free(matrix);
			block = n / size,
			start_row = block * rank,
			end_row = block * (rank + 1);
			if (rank == size - 1)
				end_row = n;
			malloc2dfloat(&matrix, end_row - start_row, 2*n);
			MPI_File_open(comm, "checkpoint.txt", MPI_MODE_RDWR|MPI_MODE_CREATE, MPI_INFO_NULL, &handle);
			MPI_File_read_at(handle, start_row*2*n*sizeof(matrix[0][0]), matrix[0], (end_row - start_row) * 2*n, MPI_DOUBLE, &status);
			MPI_Barrier(comm);
		    MPI_File_close(&handle);
        }
        MPI_Barrier(comm);
        for (int j = 0; j < size; j ++) {
			if (rank == j) {
				print_matrix(matrix, end_row - start_row, 2*n);
			}
			MPI_Barrier(comm);
	    }
	    MPI_Barrier(comm);    
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank== 0) 
    printf("\nANSWER\n");fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i ++) {
        if (rank == i) {
            print_matrix(matrix, end_row - start_row, 2*n);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
