__global__ void grid_shift(double *x , double *y , double *z , double *r, int N )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] += r[0];
        y[tid] += r[1];
        z[tid] += r[2];
    }
}
__global__ void de_grid_shift(double *x , double *y , double *z , double *r , int N )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] -= r[0];
        y[tid] -= r[1];
        z[tid] -= r[2];
    }
}

__host__ void Sort_begin(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ , double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );            

    LEBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx, ux , d_L, real_time, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx ,d_vy, d_vz, ux , d_L, real_time , N);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );



    grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx ,d_mdVy, d_mdVz, ux , d_L, real_time , Nmd);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    
    
    cellSort<<<grid_size,blockSize>>>(d_x,d_y,d_z,d_L,d_index,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cellSort<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__host__ void Sort_finish(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ ,double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    de_grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx, ux , d_L, real_time, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_vx ,d_vy, d_vz, ux , d_L, real_time , N);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    
    de_grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx ,d_mdVy, d_mdVz, ux , d_L, real_time , Nmd);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    
}

__global__ void noslip_grid_shift(double *x , double *y , double *z , double *r, int N, double *L)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] += r[0];
        y[tid] += r[1];
        z[tid] += r[2];
    }
}
__global__ void noslip_de_grid_shift(double *x , double *y , double *z , double *r , int N, double *L )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] -= r[0];
        y[tid] -= r[1];
        z[tid] -= r[2];

        if(x[tid] > L[0]/2 || x[tid]< -L[0]/2 || y[tid] > L[1]/2 || y[tid]< -L[1]/2 || z[tid] > L[2]/2 || z[tid]< -L[2]/2 ){

            printf(" after noslip degrid shift still is out x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, x[tid], tid, y[tid], tid, z[tid]);
        }
    }
}


__host__ void noslip_Sort_begin(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ , double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    noslip_grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r, N, d_L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );            

    /*noslip_BC_xyz_MPCD<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_L, real_time, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    noslip_grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd, d_L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /*noslip_BC_xyz_MD<<<grid_size, blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, real_time , Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/


    cellSort<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_L, d_index, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cellSort<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

__host__ void noslip_Sort_finish(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ ,double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r ,int N ,int Nmd , double real_time , int grid_size)
{
    noslip_de_grid_shift<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_r, N, d_L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    
    noslip_de_grid_shift<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_r, Nmd, d_L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  
    
}


__global__ void shift_index_forward(double* x, double* y, double* z, double *L, int* index, double *r, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
    index[tid] = int(x[tid] + L[0] / 2 + r[0]) + L[0] * int(y[tid] + L[1] / 2 + r[1]) + L[0] * L[1] * int(z[tid] + L[2] / 2 + r[2]);
    }

} //Output: The index array will be updated with the computed unique IDs.

__global__ void shift_index_backward(double* x,double* y,double* z, double *L, int* index, double *r, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
    index[tid] = int(x[tid] + L[0] / 2 - r[0]) + L[0] * int(y[tid] + L[1] / 2 - r[1]) + L[0] * L[1] * int(z[tid] + L[2] / 2 - r[2]);
    }

} //Output: The index array will be updated with the computed unique IDs.



//let's shift cells instead of shifting the particles

__host__ void cell_index_shift_forward(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ , double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r , int N, int Nmd, double real_time, int grid_size){

    cellSort<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_L, d_index, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cellSort<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    shift_index_forward<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_L, d_index, d_r, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    shift_index_forward<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


}


__host__ void cell_index_shift_backward(double *d_x , double *d_y , double *d_z ,double *d_vx, double *d_vy, double *d_vz, int *d_index ,
    double *d_mdX , double *d_mdY , double *d_mdZ , double *d_mdVx, double *d_mdVy, double *d_mdVz, int *d_mdIndex , double ux,
    double *d_L , double *d_r , int N, int Nmd, double real_time, int grid_size){

    shift_index_backward<<<grid_size,blockSize>>>(d_x, d_y, d_z, d_L, d_index, d_r, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    shift_index_backward<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_L, d_mdIndex, d_r, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


}
