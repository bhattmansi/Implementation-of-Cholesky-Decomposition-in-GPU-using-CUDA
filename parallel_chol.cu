#include<stdio.h>
#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include "chol.h"
int check_if_diagonal_dominant(const Matrix M);

Matrix create_positive_definite_matrix(unsigned int, unsigned int);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix A; // The N x N input matrix
Matrix h_A; // The upper triangular matrix computed by the CPU
void print_matrix(const Matrix);
int check_if_symmetric(const Matrix M); 
#define MATRIX_SIZE 3
__global__ void chol_kernel(double * U, int ops_per_thread)
{
	//Determine the boundaries for this thread
        //Get a thread identifier
        int tx = blockIdx.x * blockDim.x + threadIdx.x;

         unsigned int i, j, k;
        
        unsigned int num_rows = MATRIX_SIZE;

        //Contents of the A matrix should already be in U

        //Perform the Cholesky decomposition in place on the U matrix
        for(k = 0; k < num_rows; k++)

        {
                //Only one thread does squre root and division
                if(tx==0)
                {
                        // Take the square root of the diagonal element
                        U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
                        //Don't bother doing check...live life on the edge!

                        // Division step
                        for(j = (k + 1); j < num_rows; j++)
                        {
                                U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
                        }
                }
			
                
                __syncthreads();

	//Elimination step
	
        int istart = ( k + 1 )  +  tx * ops_per_thread;
        int iend = istart + ops_per_thread;
		 
        for (i = istart; i < iend; i++) {
            //Do work  for this i iteration
		 
            for (j = i; j < num_rows; j++) {
		 
		
                U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
            }
        }

 

                __syncthreads();
        }


        __syncthreads();


        //As the final step, zero out the lower triangular portion of U

        //Starting index for this thread
        int istart = tx * ops_per_thread;
    //Ending index for this thread
    int iend = istart + ops_per_thread;

    //Check boundaries, else do nothing
    for (i = istart; i < iend; i++) {
        //Do work  for this i iteration
        for (j = 0; j < i; j++) {
	    
            U[i * num_rows + j] = 0.0;
        
    }
    
}
}

int main()
{

        A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		print_matrix(A);
	


        int num_blocks = 1;

        //Max per block threads
        int threads_per_block = 512;

        //Operations per thread
        float ops_per_thread = MATRIX_SIZE / (threads_per_block*num_blocks);

        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks,1);

//        srand(time(NULL));


      Matrix d_A = allocate_matrix_on_gpu( A );
      

        //Copy matrices to gpu, copy A right into U
	copy_matrix_to_device( d_A, A );

        // Launch the kernel <<<grid, thread_block>>>
        chol_kernel<<<grid, thread_block>>>(d_A.elements,ops_per_thread);



       cudaDeviceSynchronize();

     copy_matrix_from_device(A, d_A);

     print_matrix(A);


// Release device memory
    cudaFree(d_A.elements);



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

	free(transpose.elements);

	// M is diagonally dominant and symmetric!
	return M;
}



Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(double);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(double);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(double);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
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

