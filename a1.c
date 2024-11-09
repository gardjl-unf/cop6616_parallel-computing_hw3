/**
 *  if (rank = 0){
 *      MPI_Barrier(…);
 *  } else {
 *      do something;
 *  }
 */

/** 
 * if (rank = 0) { should be if (rank == 0) {
 * MPI_Barrier(...) should be called by all processes, not just rank 0
 * The code as written will cause a deadlock because rank 0 will be waiting 
 * for all other processes to reach the barrier, but the other processes will 
 * not reach the barrier because they are not calling MPI_Barrier(...).
 * 
 * SOURCE:  https://rookiehpc.org/mpi/docs/mpi_barrier/index.html
 */

MPI_Barrier(...);
if (rank == 0) {
    // do a thing with rank 0 if you would like
} else {
    // do something with all other ranks
}

For a structure:
struct Particle {
    double posx, posy, posz;
    double velx, vely, velz;
    double mass;
};

SoA memory layout would look like:
posx = {0.1, 0.2, 0.3, ...}
posy = {0.4, 0.5, 0.6, ...}
posz = {0.7, 0.8, 0.9, ...}
velx = {1.0, 1.1, 1.2, ...}
vely = {1.3, 1.4, 1.5, ...}
velz = {1.6, 1.7, 1.8, ...}
mass = {2.0, 2.1, 2.2, ...}

AoS memory layout would look like:
particles = {
    {0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0},
    {0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.1},
    {0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.2},
    ...
}

The SoA layout is better because it allows for better memory access patterns.