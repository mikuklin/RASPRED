#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>

int min(int a, int b) {
    return a ? a < b : b;
}
int solve(double **matrix, int n, int i, int size, int rank, MPI_Comm comm)
{

    double lead[n * 2];
    if (i == 1 && rank == 1) {
        raise(SIGKILL);
    }
    int block = n / size,
        start_row = block * rank,
        end_row = block * (rank + 1);
    if (rank == size - 1)
        end_row = n;

    if (rank == 0) {
        double maxmod = 0;
        int pos = i;
        for (int k = i; k < n; ++k) {
            if (fabs(matrix[k][i]) > maxmod) {
                pos = k;
                maxmod = fabs(matrix[k][i]);
            }
        }
        if (pos != i) {
            for (int j = 0; j < 2 * n; j++) {
                double tmp = matrix[i][j];
                matrix[i][j] = matrix[pos][j];
                matrix[pos][j] = tmp;
            }
        }
    }
    MPI_Barrier(comm);
    int err = MPI_Bcast(&(matrix[0][0]), n*n*2, MPI_DOUBLE, 0, comm);
    if (err != MPI_SUCCESS) {
        return -1;
    }
    if (rank == min(i / block, size - 1)) {
        for (int j = 0; j < n * 2; j++) {
            lead[j] = matrix[i][j];
        }
    }
    MPI_Barrier(comm);
    err = MPI_Bcast(lead, n * 2, MPI_DOUBLE, min(i / block, size - 1), comm);
    if (err != MPI_SUCCESS) {
        return -1;
    }
    for (int j = start_row; j < end_row; j++) {
        if (j == i) {
            double d = matrix[i][i];
            for (int k = 0; k < n * 2; k++)
                matrix[j][k] /= d;
            continue;
        }
        double d = matrix[j][i] / lead[i];
        for (int k = 0; k < n * 2; k++) {
            matrix[j][k] -= d * lead[k];
        }
    }
    int *displs = (int *)malloc(size*sizeof(int));
    int *rcounts = (int *)malloc(size*sizeof(int));
    for (int k=0; k < size; ++k) {
        displs[k] = k * block * n * 2;
        if (k == size - 1) {
            rcounts[k] = (block + (n - block * size)) * n * 2;
        } else {
            rcounts[k] = block * n * 2;
        }
    }
    MPI_Barrier(comm);
    err = MPI_Gatherv(&(matrix[start_row][0]), n * 2 * (end_row - start_row), MPI_DOUBLE, &(matrix[0][0]), rcounts, displs,  MPI_DOUBLE, 0, comm);
    if (err != MPI_SUCCESS) {
        return -1;
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
int main(int argc, char **argv) {

    int rank, size;
    double start, end;
    double **matrix;
    int n = atoi(argv[1]);
    malloc2dfloat(&matrix, n, 2*n);

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    if (rank == 0) {
        printf("Input:\n"); fflush(stdout);
        FILE *output = fopen("checkpoint.txt", "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 2 * n; ++j) {
                matrix[i][j] = (j < n) ? rand() % 10 : ((j - n == i) ? 1 : 0);
                fprintf(output, "%f ", matrix[i][j]);
                printf("%f ", matrix[i][j]); fflush(stdout);
            }
            printf("\n"); fflush(stdout);
        }
        fclose(output);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&(matrix[0][0]), n * n * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int i = 0;
    while (i < n) {
        int finished;
        int result = solve(matrix, n, i, size, rank, comm);

        if (result == 0) {
            if (rank == 0) {
                printf("\nsaving checkpoint on iter %d, %d proc alive:\n", i, size); fflush(stdout);
                FILE *output = fopen("checkpoint.txt", "w");
                for (int k = 0; k < n; ++k) {
                    for (int j = 0; j < 2 * n; ++j) {
                        fprintf(output, "%f ", matrix[k][j]);
                        printf("%f ", matrix[k][j]); fflush(stdout);
                    }
                    printf("\n"); fflush(stdout);
                }
                fclose(output);
                i++;
            }
            MPI_Barrier(comm);
            MPI_Bcast(&i, 1, MPI_INT, 0, comm);
        } else {
            MPIX_Comm_shrink(comm, &comm);
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);

            if (rank == 0) {
                printf("\nloading checkpoint on iter %d, %d proc alive:\n", i, size); fflush(stdout);
                FILE *output = fopen("checkpoint.txt", "r");
                for (int k = 0; k < n; ++k) {
                    for (int j = 0; j < 2 * n; ++j) {
                        fscanf(output, "%lf", &matrix[k][j]);
                        printf("%f ", matrix[k][j]); fflush(stdout);
                        
                    }
                    printf("\n"); fflush(stdout);
                }
                fclose(output);
            }
            MPI_Barrier(comm);
            MPI_Bcast(&(matrix[0][0]), n * n * 2, MPI_DOUBLE, 0, comm);
        }
    }

    if (rank == 0) {
        printf("\n\nAnswer:\n"); fflush(stdout);
        for (int i = 0; i < n; ++i) {
            for (int j = n; j < 2 * n; ++j) {
                printf("%lf ", matrix[i][j]); fflush(stdout);
            }
            printf("\n"); fflush(stdout);
        }
    }
    MPI_Finalize();
    return 0;
}
