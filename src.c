#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "mpi.h"
                /*
                for execution in 1 to 5 // repeat each configuration 5 times – run on Prutor 5 times
                    // The below configurations will be run on Prutor when you “Execute”
                    for P (#processes) in 8, 12 [We’ll use Px=4] (note: we may test for higher process counts, ppn = Px)
                        for N (double data points per process) in 4096^2, 8192^2 (note: we may test for larger data)
                            mpirun –np P –f hostfile ./halo Px N <num_time_steps> <seed>
                */

// #pragma prutor-mpi-args: -np 12 -ppn 4
// #pragma prutor-mpi-sysargs: 4 16777216 10 7

int main( int argc, char *argv[])
{
    MPI_Init (&argc, &argv);
    int myrank,P;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
   
    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);
    
    int Px = atoi(argv[1]);
    int N = atoi(argv[2]);
    int n = sqrt((double)N);
    int steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    // int stencil = atoi(argv[5]);
    
    double data[n][n];
    srand(seed*(myrank+10));

    // Initialize data
    int i,j;
    for (i=0; i<n; i++)
    {
        for (j=0; j<n; j++)
        {
            data[i][j] = abs(rand()+(i*rand()+j*myrank))/100;
        }
    }
    int Py = P/Px;
    
    //Position of current process
    int p_i = myrank/Px;
    int p_j = myrank%Px;

    //Determine where communication is to be done
    int left = 1, right = 1, up = 1, bot = 1;
    if (p_j == 0)       left = 0;
    if (p_j == Px-1)    right = 0;
    if (p_i == 0)       up = 0;
    if (p_i == Py-1)    bot = 0;

    /////////////////
    // With Leader //
    /////////////////
    
    MPI_Comm intra_comm; //Intra node Sub Communicator
    int color = myrank/Px;
	MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &intra_comm); 

    int intra_rank; 
    MPI_Comm_rank( intra_comm , &intra_rank);

    //Leader rank is the process with intra_rank = 0
    //Send necessary (up/bottom) rows to the leader rank, so that it passes these to the required nodes.

    double buf_1[n][n];
    for(i = 0; i<n; i++){
        for (j= 0; j<n; j++){
            buf_1[i][j] = data[i][j];
        }
    }
    
    // printf("buf aft fill of rank = %d: %lf\n", myrank, buf_1[0][0]);
    
    //Store boundary points, buffers for them
    double arr_left_2[2*n] ,arr_right_2[2*n] ,arr_up_2[2*n] ,arr_bot_2[2*n];

    memset(arr_left_2, 0, 2*n*sizeof(double));
    memset(arr_right_2, 0, 2*n*sizeof(double));
    memset(arr_up_2, 0, 2*n*sizeof(double));
    memset(arr_bot_2, 0, 2*n*sizeof(double));
    double send_left_2[2*n],send_right_2[2*n],send_up_2[2*n],send_bot_2[2*n];
    double recv_left_2[2*n],recv_right_2[2*n],recv_up_2[2*n],recv_bot_2[2*n];

    double gather_buf_up[2*n*Px], gather_buf_bot[2*n*Px];
    // double *gather_buf_bot;
    double recv_gather_up[2*n*Px], recv_gather_bot[2*n*Px];
    // double scatter_recv_up[2*n],scatter_recv_bot[2*n];
    memset(gather_buf_up, 0, 2*n*sizeof(double)*Px);
    memset(gather_buf_bot, 0, 2*n*sizeof(double)*Px);

    int t = 0;
    MPI_Barrier( MPI_COMM_WORLD);
    double sTime_with_leader,eTime_with_leader;
    sTime_with_leader = MPI_Wtime();

    while(t<steps){
        
        //Step 1: Communication

        //Pack corresponding row/column and send
      
        int pos;
        if (up){
            pos = 0;
            MPI_Gather( &buf_1[0][0], 2*n, MPI_DOUBLE, gather_buf_up + intra_rank*2*n, 2*n, MPI_DOUBLE, 0, intra_comm);
            if(intra_rank == 0) MPI_Isend(gather_buf_up, pos, MPI_DOUBLE, myrank-Px, myrank, MPI_COMM_WORLD, &request);
        }
        
        if (bot){
            pos = 0;
            

            MPI_Gather( &buf_1[n-2][0], 2*n, MPI_DOUBLE, gather_buf_bot + intra_rank*2*n, 2*n, MPI_DOUBLE, 0, intra_comm);
            if(intra_rank == 0) MPI_Isend(gather_buf_bot, pos, MPI_DOUBLE, myrank+Px, myrank, MPI_COMM_WORLD, &request);
        }
        
        if(left){
            pos=0;
            for(i=0; i<n; i++) MPI_Pack(&buf_1[i][0],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_1[i][1],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);
            MPI_Isend(send_left_2,pos,MPI_PACKED,myrank-1,myrank,MPI_COMM_WORLD,&request);
        }
        if(right){
            pos=0;
            for( i=0; i<n; i++) MPI_Pack(&buf_1[i][n-1], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_1[i][n-2], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            MPI_Isend(send_right_2, pos, MPI_PACKED, myrank+1, myrank, MPI_COMM_WORLD, &request);
        }
        
        //Receiving data from neighbouring processes (Unpack and store in corresponding arrays)
        
        if (up){
            if(intra_rank == 0) MPI_Recv(recv_gather_up, 2*n*Px*sizeof(double), MPI_DOUBLE, myrank-Px, myrank-Px, MPI_COMM_WORLD, &status);
            MPI_Scatter( recv_gather_up , 2*n , MPI_DOUBLE , arr_up_2, 2*n , MPI_DOUBLE , 0, intra_comm);
        }
        
        if (bot){
            if(intra_rank == 0) MPI_Recv(recv_gather_bot, 2*n*Px*sizeof(double), MPI_DOUBLE, myrank+Px, myrank+Px, MPI_COMM_WORLD, &status);
            
            MPI_Scatter( recv_gather_bot , 2*n, MPI_DOUBLE , arr_bot_2  , 2*n, MPI_DOUBLE , 0, intra_comm);
            
        }
        
        if(left){
            pos=0;
            MPI_Recv(recv_left_2, 2*n*sizeof(double), MPI_PACKED, myrank-1, myrank-1, MPI_COMM_WORLD, &status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_left_2, 2*n*sizeof(double), &pos, arr_left_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(right){
            pos=0;
            MPI_Recv(recv_right_2, 2*n*sizeof(double), MPI_PACKED, myrank+1, myrank+1, MPI_COMM_WORLD, &status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_right_2, 2*n*sizeof(double), &pos, arr_right_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        //Step 2: Averaging
        double temp[n][n];
       
        //Corner points
        int d = 9;
        if (!left || !up) d=7; 
        if (!left && !up) d=5;
        temp[0][0]     = (arr_up_2[0] + arr_up_2[n] + arr_left_2[0] + arr_left_2[n] + buf_1[0][1] + buf_1[0][2] + buf_1[1][0]+ buf_1[2][0] + buf_1[0][0])/d;
        d = 9;
        if (!right || !up) d= 7;  
        if (!right && !up) d= 5; 
        temp[0][n-1]   = (arr_up_2[n-1] + arr_up_2[2*n-1] + arr_right_2[0] + arr_right_2[n] + buf_1[0][n-2] + buf_1[0][n-3] + buf_1[1][n-1] + buf_1[2][n-1] + buf_1[0][n-1])/d;
        d = 9;
        if (!left || !bot) d= 7;
        if (!left && !bot) d= 5;
        temp[n-1][0]   = (arr_bot_2[0] + arr_bot_2[n] + arr_left_2[n-1]+arr_left_2[2*n-1] + buf_1[n-1][1] + buf_1[n-2][0] + buf_1[n-3][0] + buf_1[n-1][2] + buf_1[n-1][0])/d;
        d = 9;
        if (!bot || !right) d = 7; 
        if (!bot && !right) d = 5; 
        temp[n-1][n-1] = (arr_bot_2[n-1] + arr_bot_2[2*n-1] + arr_right_2[n-1] + arr_right_2[2*n-1] + buf_1[n-2][n-1] + buf_1[n-3][n-1] + buf_1[n-1][n-2] + buf_1[n-1][n-3] + buf_1[n-1][n-1])/d;  
        d = 9;

        //Second to Corner
        if (!left || !up) d=8; 
        if (!left && !up) d=7;
        temp[1][1]     = (buf_1[0][1] + buf_1[2][1] + buf_1[3][1] + buf_1[1][0] + buf_1[1][2] + buf_1[1][3] + arr_up_2[1] + arr_left_2[1] + buf_1[1][1])/d;
        d = 9;
        if (!right || !up) d= 8;  
        if (!right && !up) d= 7; 
        temp[1][n-2]   = (buf_1[0][n-2] + buf_1[2][n-2] + buf_1[3][n-2] + buf_1[1][n-4] + buf_1[1][n-3] + buf_1[1][n-1] + arr_up_2[n-2] + arr_right_2[1] + buf_1[1][n-2])/d;
        d = 9;
        if (!left || !bot) d= 8;
        if (!left && !bot) d= 7;
        temp[n-2][1]   = (buf_1[n-4][1] + buf_1[n-3][1] + buf_1[n-1][1] + buf_1[n-2][0] + buf_1[n-2][2] + buf_1[n-2][3] + arr_bot_2[1] + arr_left_2[n-2] + buf_1[n-2][1])/d;
        d = 9;
        if (!bot || !right) d = 8; 
        if (!bot && !right) d = 7; 
        temp[n-2][n-2] = (buf_1[n-2][n-1] + buf_1[n-2][n-3] + buf_1[n-2][n-4] + buf_1[n-1][n-2] + buf_1[n-3][n-2] + buf_1[n-4][n-2] + arr_bot_2[n-2] + arr_right_2[n-2] + buf_1[n-2][n-2])/d;
        d = 9;

        if (!up) d = 7;
        if (!left) d = 8;
        if (!left && !up) d = 6;
        temp[0][1]     = (buf_1[0][2] + buf_1[0][3] + buf_1[0][0] + buf_1[1][1] + buf_1[2][1] + arr_left_2[0] + arr_up_2[1] + arr_up_2[1+n] + buf_1[0][1])/d;
        d = 9;
        if (!up) d = 8;
        if (!left) d = 7;
        if (!left && !up) d = 6;
        temp[1][0]     = (buf_1[0][0] + buf_1[2][0] + buf_1[3][0] + buf_1[1][2] + buf_1[1][1]+arr_up_2[0] + arr_left_2[1] + arr_left_2[1+n] + buf_1[1][0])/d;
        d = 9;
        if (!up) d = 7;
        if (!right) d = 8;
        if (!right && !up) d = 6;
        temp[0][n-2]   = (buf_1[1][n-2] + buf_1[2][n-2] + buf_1[0][n-4] + buf_1[0][n-3] + buf_1[0][n-1] + arr_up_2[2*n-2] + arr_up_2[n-2] + arr_right_2[0] + buf_1[0][n-2])/d;
        d = 9;
        if (!up) d = 8;
        if (!right) d = 7;
        if (!right && !up) d = 6;
        
        temp[1][n-1]   = (buf_1[0][n-1] + buf_1[2][n-1] + buf_1[3][n-1] + buf_1[1][n-2] + buf_1[1][n-3] + arr_up_2[n-1] + arr_right_2[1+n] + arr_right_2[1] + buf_1[1][n-1])/d;
        d = 9;
        if (!bot) d = 8;
        if (!left) d = 7;
        if (!left && !bot) d = 6;
        temp[n-2][0]   = (buf_1[n-4][0] + buf_1[n-3][0] + buf_1[n-1][0] + buf_1[n-2][1] + buf_1[n-2][2] + arr_bot_2[0] + arr_left_2[n-2] + arr_left_2[2*n-2] + buf_1[n-2][0])/d;
        d = 9;
        if (!bot) d = 7;
        if (!left) d = 8;
        if (!left && !bot) d = 6;
        temp[n-1][1]   = (buf_1[n-2][1] + buf_1[n-3][1] + buf_1[n-1][0] + buf_1[n-1][3] + buf_1[n-1][2] + arr_bot_2[1] + arr_bot_2[n+1] + arr_left_2[n-1] + buf_1[n-1][1])/d;
        d = 9;
        if (!bot) d = 8;
        if (!right) d = 7;
        if (!right && !bot) d = 6;
        temp[n-2][n-1] = (buf_1[n-2][n-3] + buf_1[n-2][n-2] + buf_1[n-1][n-1] + buf_1[n-3][n-1] + buf_1[n-4][n-1] + arr_bot_2[n-1] + arr_right_2[n-2] + arr_right_2[2*n-2] + buf_1[n-2][n-1])/d;
        d = 9;
        if (!bot) d = 7;
        if (!right) d = 8;
        if (!right && !bot) d = 6;
        temp[n-1][n-2] = (buf_1[n-1][n-3] + buf_1[n-1][n-4] + buf_1[n-1][n-1] + buf_1[n-3][n-2] + buf_1[n-2][n-2] + arr_bot_2[n-2] + arr_bot_2[2*n-2] + arr_right_2[n-1] + buf_1[n-1][n-2])/d;
        d = 9;

        //Edges
        for( i=2; i<n-2; i++)
        {
            d = 9;
            if(!up) d = 7;
            temp[0][i]   = (buf_1[0][i+1] + buf_1[0][i+2] + buf_1[0][i-1] + buf_1[0][i-2] + buf_1[1][i] + buf_1[2][i] + arr_up_2[i] + arr_up_2[i+n] + buf_1[0][i])/d;
            d = 9;
            if(!up) d = 8;
            temp[1][i]   = (buf_1[1][i+1] + buf_1[1][i+2] + buf_1[1][i-1] + buf_1[1][i-2] + buf_1[0][i] + buf_1[2][i] + buf_1[3][i] + arr_up_2[i] + buf_1[1][i])/d;
            d = 9;
            if(!bot) d = 7;
            temp[n-1][i] = (buf_1[n-1][i+1] + buf_1[n-1][i+2] + buf_1[n-1][i-1]+buf_1[n-1][i-2] + buf_1[n-3][i]+ buf_1[n-2][i] + arr_bot_2[i]+arr_bot_2[i+n]+ buf_1[n-1][i])/d;
            d = 9;
            if(!bot) d = 8;
            temp[n-2][i] = (buf_1[n-2][i+1] + buf_1[n-2][i+2] + buf_1[n-2][i-1]+buf_1[n-2][i-2] + buf_1[n-1][i] + buf_1[n-3][i] + buf_1[n-4][i] + arr_bot_2[i] + buf_1[n-2][i])/d;
            d = 9;
            if(!left) d = 7;
            temp[i][0]   = (buf_1[i+1][0]+buf_1[i+2][0]+buf_1[i-2][0] + buf_1[i-1][0] + buf_1[i][1]+buf_1[i][2] + arr_left_2[i]+ arr_left_2[i+n] + buf_1[i][0])/d;
            d = 9;
            if(!left) d = 8;
            temp[i][1]   = (buf_1[i+1][1]+buf_1[i+2][1]+buf_1[i-2][1] + buf_1[i-1][1] + buf_1[i][0]+buf_1[i][2] + buf_1[i][3] + arr_left_2[i] + buf_1[i][1])/d;
            d = 9;
            if(!right) d = 7;
            temp[i][n-1] = (buf_1[i+1][n-1]+buf_1[i+2][n-1] + buf_1[i-2][n-1]+ buf_1[i-1][n-1] + buf_1[i][n-2] +buf_1[i][n-3]+ arr_right_2[i] + arr_right_2[i+n]+ buf_1[i][n-1])/d;
            d = 9;
            if(!right) d = 8;
            temp[i][n-2] = (buf_1[i+1][n-2]+buf_1[i+2][n-2] + buf_1[i-2][n-2]+ buf_1[i-1][n-2] + buf_1[i][n-3] +  buf_1[i][n-1]+buf_1[i][n-4]+ + arr_right_2[i] + buf_1[i][n-2])/d;
            d = 9;
        }

        //Interior points
        for(i=2;i<n-2;i++){
            for(j=2;j<n-2;j++)
            {
                temp[i][j] = (buf_1[i-1][j] + buf_1[i-2][j] + buf_1[i+1][j] + buf_1[i+2][j] + buf_1[i][j-1] + buf_1[i][j-2] + buf_1[i][j+1] + buf_1[i][j+2] + buf_1[i][j])/9;
            }
        }
        
        //Incrementing the steps, copying temporary data to original matrix
        for(i = 0; i<n; i++){
            for (j= 0; j<n; j++){
                buf_1[i][j] = temp[i][j];
            }
        }
        
        t++;
    }
    eTime_with_leader = MPI_Wtime();
    double time_with_leader = eTime_with_leader-sTime_with_leader;
    double maxTime_with_leader;

    //Get maximum of time taken by all processes
    MPI_Reduce (&time_with_leader, &maxTime_with_leader, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    

    ////////////////////
    // Without Leader //
    ////////////////////
    double buf_0[n][n];
    for( i = 0; i<n; i++){
        for ( j= 0; j<n; j++){
            buf_0[i][j] = data[i][j];
        }
    }

    //Store boundary points, buffers for them
    // double arr_left_2[2*n] ,arr_right_2[2*n] ,arr_up_2[2*n] ,arr_bot_2[2*n];

    memset( arr_left_2, 0, 2*n*sizeof(double) );
    memset( arr_right_2, 0, 2*n*sizeof(double) );
    memset( arr_up_2, 0, 2*n*sizeof(double) );
    memset( arr_bot_2, 0, 2*n*sizeof(double) );
    // double send_left_2[2*n],send_right_2[2*n],send_up_2[2*n],send_bot_2[2*n];
    // double recv_left_2[2*n],recv_right_2[2*n],recv_up_2[2*n],recv_bot_2[2*n];

    t = 0;
    MPI_Barrier( MPI_COMM_WORLD);
    double sTime_no_leader, eTime_no_leader;
    sTime_no_leader = MPI_Wtime();

    while(t<steps){
        
        //Step 1: Communication

        //Pack corresponding row/column and send
        int pos;
        if(up){
            pos = 0;
            for(i=0; i<n; i++) MPI_Pack(&buf_0[0][i], 1, MPI_DOUBLE, send_up_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_0[1][i], 1, MPI_DOUBLE, send_up_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            MPI_Isend(send_up_2, pos, MPI_PACKED, myrank-Px, myrank, MPI_COMM_WORLD, &request);
        }
        if(bot){
            pos=0;
            for( i=0; i<n; i++) MPI_Pack(&buf_0[n-1][i], 1, MPI_DOUBLE, send_bot_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_0[n-2][i], 1, MPI_DOUBLE, send_bot_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            MPI_Isend(send_bot_2, pos, MPI_PACKED, myrank+Px, myrank, MPI_COMM_WORLD, &request); 
        }
        if(left){
            pos=0;
            for( i=0; i<n; i++) MPI_Pack(&buf_0[i][0],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_0[i][1],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);
            MPI_Isend(send_left_2,pos,MPI_PACKED,myrank-1,myrank,MPI_COMM_WORLD,&request);
        }
        if(right){
            pos=0;
            for( i=0; i<n; i++) MPI_Pack(&buf_0[i][n-1], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            for( i=0; i<n; i++) MPI_Pack(&buf_0[i][n-2], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
            MPI_Isend(send_right_2, pos, MPI_PACKED, myrank+1, myrank, MPI_COMM_WORLD, &request);
        }
        
        //Receiving data from neighbouring processes (Unpack and store in corresponding arrays)
        if(up){
            pos=0;
            MPI_Recv(recv_up_2, 2*n*sizeof(double), MPI_PACKED, myrank-Px, myrank-Px, MPI_COMM_WORLD, &status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_up_2, 2*n*sizeof(double), &pos, arr_up_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(bot){
            pos=0;
            MPI_Recv(recv_bot_2,2*n*sizeof(double),MPI_PACKED,myrank+Px,myrank+Px,MPI_COMM_WORLD,&status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_bot_2, 2*n*sizeof(double), &pos, arr_bot_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(left){
            pos=0;
            MPI_Recv(recv_left_2, 2*n*sizeof(double), MPI_PACKED, myrank-1, myrank-1, MPI_COMM_WORLD, &status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_left_2, 2*n*sizeof(double), &pos, arr_left_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(right){
            pos=0;
            MPI_Recv(recv_right_2, 2*n*sizeof(double), MPI_PACKED, myrank+1, myrank+1, MPI_COMM_WORLD, &status);
            for( i=0; i<2*n; i++) MPI_Unpack(recv_right_2, 2*n*sizeof(double), &pos, arr_right_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }

        //Step 2: Averaging
        double temp[n][n];
        
        //Corner points
        int d = 9;
        if (!left || !up) d=7; 
        if (!left && !up) d=5;
        temp[0][0]     = (arr_up_2[0] + arr_up_2[n] + arr_left_2[0] + arr_left_2[n] + buf_0[0][1] + buf_0[0][2] + buf_0[1][0]+ buf_0[2][0] + buf_0[0][0])/d;
        d = 9;
        if (!right || !up) d= 7;  
        if (!right && !up) d= 5; 
        temp[0][n-1]   = (arr_up_2[n-1] + arr_up_2[2*n-1] + arr_right_2[0] + arr_right_2[n] + buf_0[0][n-2] + buf_0[0][n-3] + buf_0[1][n-1] + buf_0[2][n-1] + buf_0[0][n-1])/d;
        d = 9;
        if (!left || !bot) d= 7;
        if (!left && !bot) d= 5;
        temp[n-1][0]   = (arr_bot_2[0] + arr_bot_2[n] + arr_left_2[n-1]+arr_left_2[2*n-1] + buf_0[n-1][1] + buf_0[n-2][0] + buf_0[n-3][0] + buf_0[n-1][2] + buf_0[n-1][0])/d;
        d = 9;
        if (!bot || !right) d = 7; 
        if (!bot && !right) d = 5; 
        temp[n-1][n-1] = (arr_bot_2[n-1] + arr_bot_2[2*n-1] + arr_right_2[n-1] + arr_right_2[2*n-1] + buf_0[n-2][n-1] + buf_0[n-3][n-1] + buf_0[n-1][n-2] + buf_0[n-1][n-3] + buf_0[n-1][n-1])/d;  
        d = 9;

        //Second to Corner
        if (!left || !up) d=8; 
        if (!left && !up) d=7;
        temp[1][1]     = (buf_0[0][1] + buf_0[2][1] + buf_0[3][1] + buf_0[1][0] + buf_0[1][2] + buf_0[1][3] + arr_up_2[1] + arr_left_2[1] + buf_0[1][1])/d;
        d = 9;
        if (!right || !up) d= 8;  
        if (!right && !up) d= 7; 
        temp[1][n-2]   = (buf_0[0][n-2] + buf_0[2][n-2] + buf_0[3][n-2] + buf_0[1][n-4] + buf_0[1][n-3] + buf_0[1][n-1] + arr_up_2[n-2] + arr_right_2[1] + buf_0[1][n-2])/d;
        d = 9;
        if (!left || !bot) d= 8;
        if (!left && !bot) d= 7;
        temp[n-2][1]   = (buf_0[n-4][1] + buf_0[n-3][1] + buf_0[n-1][1] + buf_0[n-2][0] + buf_0[n-2][2] + buf_0[n-2][3] + arr_bot_2[1] + arr_left_2[n-2] + buf_0[n-2][1])/d;
        d = 9;
        if (!bot || !right) d = 8; 
        if (!bot && !right) d = 7; 
        temp[n-2][n-2] = (buf_0[n-2][n-1] + buf_0[n-2][n-3] + buf_0[n-2][n-4] + buf_0[n-1][n-2] + buf_0[n-3][n-2] + buf_0[n-4][n-2] + arr_bot_2[n-2] + arr_right_2[n-2] + buf_0[n-2][n-2])/d;
        d = 9;

        if (!up) d = 7;
        if (!left) d = 8;
        if (!left && !up) d = 6;
        temp[0][1]     = (buf_0[0][2] + buf_0[0][3] + buf_0[0][0] + buf_0[1][1] + buf_0[2][1] + arr_left_2[0] + arr_up_2[1] + arr_up_2[1+n] + buf_0[0][1])/d;
        d = 9;
        if (!up) d = 8;
        if (!left) d = 7;
        if (!left && !up) d = 6;
        temp[1][0]     = (buf_0[0][0] + buf_0[2][0] + buf_0[3][0] + buf_0[1][2] + buf_0[1][1]+arr_up_2[0] + arr_left_2[1] + arr_left_2[1+n] + buf_0[1][0])/d;
        d = 9;
        if (!up) d = 7;
        if (!right) d = 8;
        if (!right && !up) d = 6;
        temp[0][n-2]   = (buf_0[1][n-2] + buf_0[2][n-2] + buf_0[0][n-4] + buf_0[0][n-3] + buf_0[0][n-1] + arr_up_2[2*n-2] + arr_up_2[n-2] + arr_right_2[0] + buf_0[0][n-2])/d;
        d = 9;
        if (!up) d = 8;
        if (!right) d = 7;
        if (!right && !up) d = 6;
        
        temp[1][n-1]   = (buf_0[0][n-1] + buf_0[2][n-1] + buf_0[3][n-1] + buf_0[1][n-2] + buf_0[1][n-3] + arr_up_2[n-1] + arr_right_2[1+n] + arr_right_2[1] + buf_0[1][n-1])/d;
        d = 9;
        if (!bot) d = 8;
        if (!left) d = 7;
        if (!left && !bot) d = 6;
        temp[n-2][0]   = (buf_0[n-4][0] + buf_0[n-3][0] + buf_0[n-1][0] + buf_0[n-2][1] + buf_0[n-2][2] + arr_bot_2[0] + arr_left_2[n-2] + arr_left_2[2*n-2] + buf_0[n-2][0])/d;
        d = 9;
        if (!bot) d = 7;
        if (!left) d = 8;
        if (!left && !bot) d = 6;
        temp[n-1][1]   = (buf_0[n-2][1] + buf_0[n-3][1] + buf_0[n-1][0] + buf_0[n-1][3] + buf_0[n-1][2] + arr_bot_2[1] + arr_bot_2[n+1] + arr_left_2[n-1] + buf_0[n-1][1])/d;
        d = 9;
        if (!bot) d = 8;
        if (!right) d = 7;
        if (!right && !bot) d = 6;
        temp[n-2][n-1] = (buf_0[n-2][n-3] + buf_0[n-2][n-2] + buf_0[n-1][n-1] + buf_0[n-3][n-1] + buf_0[n-4][n-1] + arr_bot_2[n-1] + arr_right_2[n-2] + arr_right_2[2*n-2] + buf_0[n-2][n-1])/d;
        d = 9;
        if (!bot) d = 7;
        if (!right) d = 8;
        if (!right && !bot) d = 6;
        temp[n-1][n-2] = (buf_0[n-1][n-3] + buf_0[n-1][n-4] + buf_0[n-1][n-1] + buf_0[n-3][n-2] + buf_0[n-2][n-2] + arr_bot_2[n-2] + arr_bot_2[2*n-2] + arr_right_2[n-1] + buf_0[n-1][n-2])/d;
        d = 9;

        //Edges
        for( i=2; i<n-2; i++)
        {
            d = 9;
            if(!up) d = 7;
            temp[0][i]   = (buf_0[0][i+1] + buf_0[0][i+2] + buf_0[0][i-1] + buf_0[0][i-2] + buf_0[1][i] + buf_0[2][i] + arr_up_2[i] + arr_up_2[i+n] + buf_0[0][i])/d;
            d = 9;
            if(!up) d = 8;
            temp[1][i]   = (buf_0[1][i+1] + buf_0[1][i+2] + buf_0[1][i-1] + buf_0[1][i-2] + buf_0[0][i] + buf_0[2][i] + buf_0[3][i] + arr_up_2[i] + buf_0[1][i])/d;
            d = 9;
            if(!bot) d = 7;
            temp[n-1][i] = (buf_0[n-1][i+1] + buf_0[n-1][i+2] + buf_0[n-1][i-1]+buf_0[n-1][i-2] + buf_0[n-3][i]+ buf_0[n-2][i] + arr_bot_2[i]+arr_bot_2[i+n]+ buf_0[n-1][i])/d;
            d = 9;
            if(!bot) d = 8;
            temp[n-2][i] = (buf_0[n-2][i+1] + buf_0[n-2][i+2] + buf_0[n-2][i-1]+buf_0[n-2][i-2] + buf_0[n-1][i] + buf_0[n-3][i] + buf_0[n-4][i] + arr_bot_2[i] + buf_0[n-2][i])/d;
            d = 9;
            if(!left) d = 7;
            temp[i][0]   = (buf_0[i+1][0]+buf_0[i+2][0]+buf_0[i-2][0] + buf_0[i-1][0] + buf_0[i][1]+buf_0[i][2] + arr_left_2[i]+ arr_left_2[i+n] + buf_0[i][0])/d;
            d = 9;
            if(!left) d = 8;
            temp[i][1]   = (buf_0[i+1][1]+buf_0[i+2][1]+buf_0[i-2][1] + buf_0[i-1][1] + buf_0[i][0]+buf_0[i][2] + buf_0[i][3] + arr_left_2[i] + buf_0[i][1])/d;
            d = 9;
            if(!right) d = 7;
            temp[i][n-1] = (buf_0[i+1][n-1]+buf_0[i+2][n-1] + buf_0[i-2][n-1]+ buf_0[i-1][n-1] + buf_0[i][n-2] +buf_0[i][n-3]+ arr_right_2[i] + arr_right_2[i+n]+ buf_0[i][n-1])/d;
            d = 9;
            if(!right) d = 8;
            temp[i][n-2] = (buf_0[i+1][n-2]+buf_0[i+2][n-2] + buf_0[i-2][n-2]+ buf_0[i-1][n-2] + buf_0[i][n-3] +  buf_0[i][n-1]+buf_0[i][n-4]+ + arr_right_2[i] + buf_0[i][n-2])/d;
            d = 9;
        }

        //Interior points
        for(i=2;i<n-2;i++){
            for( j=2;j<n-2;j++)
            {
                temp[i][j] = (buf_0[i-1][j] + buf_0[i-2][j] + buf_0[i+1][j] + buf_0[i+2][j] + buf_0[i][j-1] + buf_0[i][j-2] + buf_0[i][j+1] + buf_0[i][j+2] + buf_0[i][j])/9;
            }
        }
        
        //Incrementing the steps, copying temporary data to original matrix
        t++;
        for(i = 0; i<n; i++){
            for (j= 0; j<n; j++){
                buf_0[i][j] = temp[i][j];
            }
        }
        
    }
    eTime_no_leader = MPI_Wtime();
    double time_no_leader = eTime_no_leader-sTime_no_leader;
    double maxTime_no_leader;

    //Get maximum of time taken by all processes
    MPI_Reduce (&time_no_leader, &maxTime_no_leader, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(myrank == 0){
        printf("Processes: %d, Data size: %d\n",P, n);
        printf("Time with leader: %lf\n",maxTime_with_leader);
        printf("Time without leader: %lf\n",maxTime_no_leader);
        printf("Data: %lf\n",buf_1[0][0]);
    }

    MPI_Finalize();
    return 0;
}
