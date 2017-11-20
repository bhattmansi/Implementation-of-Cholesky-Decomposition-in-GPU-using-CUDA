
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include "chol.h"
int check_if_diagonal_dominant(const Matrix M);

Matrix create_positive_definite_matrix(unsigned int, unsigned int);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
Matrix A; // The N x N input matrix
void print_matrix(const Matrix);
int check_if_symmetric(const Matrix M); 
#define MATRIX_SIZE 3


int main()
{
	int n = MATRIX_SIZE;
        A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		print_matrix(A);
	

        Matrix L = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
 
        for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L.elements[i * n + k] * L.elements[j * n + k];
            L.elements[i * n + j] = (i == j) ? sqrt(A.elements[i * n + i] - s) : (1.0 / L.elements[j * n + j] * (A.elements[i * n + j] - s));
        }

      
       print_matrix(L);


    // Release host memory
   free(A.elements);


 return 0;


}
// matrix M is positive definite if and only if the determinant of each of the principal submatrices is positive. 
  // A diagonally dominant NxN symmetric matrix is positive definite. This function generates a diagonally dominant NxN symmetric matrix. */

Matrix create_positive_definite_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (double *)malloc(size * sizeof(double));

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
	printf("Creating a %d x %d matrix with random numbers between [-.5, .5]...", num_rows, num_columns);
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.elements[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
       	printf("done. \n");
	// print_matrix(M);
	// getchar();

	// Step 2: Make the matrix symmetric by adding its transpose to itself
	printf("Generating the symmetric matrix...");
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.elements = (double *)malloc(size * sizeof(double));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
	// print_matrix(transpose);

	for(i = 0; i < size; i++)
		M.elements[i] += transpose.elements[i];
	if(check_if_symmetric(M))
		printf("done. \n");
	else{ 
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// print_matrix(M);
	// getchar();

	// Step 3: Make the diagonal entries large with respect to the row and column entries
	printf("Generating the positive definite matrix...");
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.elements[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		printf("done. \n");
	else{
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// print_matrix(M);
	// getchar();

	free(transpose.elements);

	// M is diagonally dominant and symmetric!
	return M;
}


Matrix allocate_matrix(int num_rows, int num_columns, int init) {
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (double *) malloc(size * sizeof (double));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.elements[i] = 0;
        else
            M.elements[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}


void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.elements[i * M.num_rows + j] != M.elements[j * M.num_columns + i])
				return 0;
	return 1;
}

int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		if(diag_element <= sum)
			return 0;
	}

	return 1;
}

