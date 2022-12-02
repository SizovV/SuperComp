#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

int nx_low, nx_high, ny_low, ny_high, nz_low, nz_high, split_type, bmax, bnumx, bnumy, bnumz, X, Y,
    Z;
double dx, dy, dz, Lx, Ly, Lz, T = 1, dt;

double u(double x, double y, double z, double t) {
    double a_t = M_PI * sqrt(9 / (Lx * Lx) + 4 / (Ly * Ly) + 4 / (Lz * Lz));
    return sin(3 * M_PI / Lx * x) * sin(2 * M_PI / Ly * y) * sin(2 * M_PI / Lz * z) *
           cos(a_t * t + 4 * M_PI);
}

double phi(double x, double y, double z) { return u(x, y, z, 0); }

double ***allocate3darray(int X, int Y, int Z) {
    double ***p = new double **[X];
    for (int x = 0; x < X; x++) {
        p[x] = new double *[Y];
        for (int y = 0; y < Y; y++) {
            p[x][y] = new double[Z];
        }
    }
    return p;
}

void deallocate3darray(double ***p, int X, int Y, int Z) {
    for (int x = 0; x < X; x++) {
        for (int y = 0; y < Y; y++) {
            delete[] p[x][y];
        }
        delete[] p[x];
    }
    delete[] p;
}

void setup_range(int rank, int N, int num_processes) {
    int cubert = round(cbrt(num_processes));
    if (cubert * cubert * cubert == num_processes) {
        // cube
        bnumx = rank / (cubert * cubert);
        bnumy = rank % (cubert * cubert) / cubert;
        bnumz = rank % cubert;
        nx_low = N / cubert * bnumx;
        nx_high = bnumx == cubert - 1 ? N : N / cubert * (bnumx + 1);
        ny_low = N / cubert * bnumy;
        ny_high = bnumy == cubert - 1 ? N : N / cubert * (bnumy + 1);
        nz_low = N / cubert * bnumz;
        nz_high = bnumz == cubert - 1 ? N : N / cubert * (bnumz + 1);
        split_type = 0;
        bmax = cubert;
    } else {
        // line
        bnumx = rank;
        bnumy = bnumz = 0;
        nx_low = N / num_processes * bnumx;
        nx_high = bnumx == num_processes - 1 ? N : N / num_processes * (bnumx + 1);
        ny_low = 0;
        ny_high = N;
        nz_low = 0;
        nz_high = N;
        split_type = 1;
        bmax = num_processes;
    }
    X = nx_high - nx_low;
    Y = ny_high - ny_low;
    Z = nz_high - nz_low;
}

int get_rank(int bnumx, int bnumy, int bnumz) {
    if (split_type == 0) {
        return bnumz + bnumy * bmax + bnumx * bmax * bmax;
    } else {
        return bnumx;
    }
}

void sendborder(double ***u_n, double *u_n_x_left_send, double *u_n_y_left_send,
                double *u_n_z_left_send, double *u_n_x_left_recv, double *u_n_y_left_recv,
                double *u_n_z_left_recv, double *u_n_x_right_send, double *u_n_y_right_send,
                double *u_n_z_right_send, double *u_n_x_right_recv, double *u_n_y_right_recv,
                double *u_n_z_right_recv) {
    MPI_Request requests[12];
    MPI_Status statuses[12];
    int num_requests = 0;

    if (bnumx != 0) {
#pragma omp parallel for
        for (int ny = 0; ny < Y; ny++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                u_n_x_left_send[ny * Z + nz] = u_n[0][ny][nz];
            }
        }
        MPI_Isend(u_n_x_left_send, Y * Z, MPI_DOUBLE, get_rank(bnumx - 1, bnumy, bnumz), 0,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_x_left_recv, Y * Z, MPI_DOUBLE, get_rank(bnumx - 1, bnumy, bnumz), 1,
                  MPI_COMM_WORLD, &requests[num_requests++]);
    }

    if (bnumx != bmax - 1) {
#pragma omp parallel for
        for (int ny = 0; ny < Y; ny++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                u_n_x_right_send[ny * Z + nz] = u_n[X - 1][ny][nz];
            }
        }
        MPI_Isend(u_n_x_right_send, Y * Z, MPI_DOUBLE, get_rank(bnumx + 1, bnumy, bnumz), 1,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_x_right_recv, Y * Z, MPI_DOUBLE, get_rank(bnumx + 1, bnumy, bnumz), 0,
                  MPI_COMM_WORLD, &requests[num_requests++]);
    }

    if (split_type == 0) {
        int ind, bind;

        // y_left
        ind = bnumy == 0 ? 1 : 0;
        bind = bnumy == 0 ? bmax - 1 : bnumy - 1;
#pragma omp parallel for
        for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                u_n_y_left_send[nx * Z + nz] = u_n[nx][ind][nz];
            }
        }
        MPI_Isend(u_n_y_left_send, X * Z, MPI_DOUBLE, get_rank(bnumx, bind, bnumz), 2,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_y_left_recv, X * Z, MPI_DOUBLE, get_rank(bnumx, bind, bnumz), 3,
                  MPI_COMM_WORLD, &requests[num_requests++]);

        // y_right
        ind = bnumy == bmax - 1 ? Y - 2 : Y - 1;
        bind = bnumy == bmax - 1 ? 0 : bnumy + 1;
#pragma omp parallel for
        for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                u_n_y_right_send[nx * Z + nz] = u_n[nx][ind][nz];
            }
        }
        MPI_Isend(u_n_y_right_send, X * Z, MPI_DOUBLE, get_rank(bnumx, bind, bnumz), 3,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_y_right_recv, X * Z, MPI_DOUBLE, get_rank(bnumx, bind, bnumz), 2,
                  MPI_COMM_WORLD, &requests[num_requests++]);

        // z_left
        ind = bnumz == 0 ? 1 : 0;
        bind = bnumz == 0 ? bmax - 1 : bnumz - 1;
#pragma omp parallel for
        for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
            for (int ny = 0; ny < Y; ny++) {
                u_n_z_left_send[nx * Y + ny] = u_n[nx][ny][ind];
            }
        }
        MPI_Isend(u_n_z_left_send, X * Y, MPI_DOUBLE, get_rank(bnumx, bnumy, bind), 4,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_z_left_recv, X * Y, MPI_DOUBLE, get_rank(bnumx, bnumy, bind), 5,
                  MPI_COMM_WORLD, &requests[num_requests++]);

        // z_right
        ind = bnumz == bmax - 1 ? Z - 2 : Z - 1;
        bind = bnumz == bmax - 1 ? 0 : bnumz + 1;
#pragma omp parallel for
        for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
            for (int ny = 0; ny < Y; ny++) {
                u_n_z_right_send[nx * Y + ny] = u_n[nx][ny][ind];
            }
        }
        MPI_Isend(u_n_z_right_send, X * Y, MPI_DOUBLE, get_rank(bnumx, bnumy, bind), 5,
                  MPI_COMM_WORLD, &requests[num_requests++]);
        MPI_Irecv(u_n_z_right_recv, X * Y, MPI_DOUBLE, get_rank(bnumx, bnumy, bind), 4,
                  MPI_COMM_WORLD, &requests[num_requests++]);
    }

    MPI_Waitall(num_requests, requests, statuses);
}

double nabla(double u_n_x_y_z, double u_n_x_left, double u_n_x_right, double u_n_y_left,
             double u_n_y_right, double u_n_z_left, double u_n_z_right) {
    return (u_n_x_left - 2 * u_n_x_y_z + u_n_x_right) / (dx * dx) +
           (u_n_y_left - 2 * u_n_x_y_z + u_n_y_right) / (dy * dy) +
           (u_n_z_left - 2 * u_n_x_y_z + u_n_z_right) / (dz * dz);
}

void initialize(double ***u_n, double ***u_n_1, double *u_n_x_left_send, double *u_n_y_left_send,
                double *u_n_z_left_send, double *u_n_x_left_recv, double *u_n_y_left_recv,
                double *u_n_z_left_recv, double *u_n_x_right_send, double *u_n_y_right_send,
                double *u_n_z_right_send, double *u_n_x_right_recv, double *u_n_y_right_recv,
                double *u_n_z_right_recv, double dt) {
    // compute u_0
#pragma omp parallel for
    for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
        for (int ny = 0; ny < Y; ny++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                double x, y, z;
                x = nx_low * dx + nx * dx;
                y = ny_low * dy + ny * dy;
                z = nz_low * dz + nz * dz;
                u_n[nx][ny][nz] = phi(x, y, z);
            }
        }
    }
    // send border u_0 values
    sendborder(u_n, u_n_x_left_send, u_n_y_left_send, u_n_z_left_send, u_n_x_left_recv,
               u_n_y_left_recv, u_n_z_left_recv, u_n_x_right_send, u_n_y_right_send,
               u_n_z_right_send, u_n_x_right_recv, u_n_y_right_recv, u_n_z_right_recv);

#pragma omp parallel for
    for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
        for (int ny = 0; ny < Y; ny++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                double x, y, z;
                double u_n_x_left, u_n_y_left, u_n_z_left, u_n_x_right, u_n_y_right, u_n_z_right;
                x = nx_low * dx + nx * dx;
                y = ny_low * dy + ny * dy;
                z = nz_low * dz + nz * dz;
                
                if (nx == 0) {
                    if (bnumx == 0) {
                        u_n_1[nx][ny][nz] = 0;
                        continue;
                    } else {
                        u_n_x_left = u_n_x_left_recv[ny * Z + nz];
                    }
                } else {
                    u_n_x_left = u_n[nx - 1][ny][nz];
                }

                if (nx == X - 1) {
                    if (bnumx == bmax - 1) {
                        u_n_1[nx][ny][nz] = 0;
                        continue;
                    } else {
                        u_n_x_right = u_n_x_right_recv[ny * Z + nz];
                    }
                } else {
                    u_n_x_right = u_n[nx + 1][ny][nz];
                }

                if (ny == 0) {
                    if (split_type == 0) {
                        u_n_y_left = u_n_y_left_recv[nx * Z + nz];
                    } else {
                        u_n_y_left = u_n[nx][Y - 2][nz];
                    }
                } else {
                    u_n_y_left = u_n[nx][ny - 1][nz];
                }

                if (ny == Y - 1) {
                    if (split_type == 0) {
                        u_n_y_right = u_n_y_right_recv[nx * Z + nz];
                    } else {
                        u_n_y_right = u_n[nx][1][nz];
                    }
                } else {
                    u_n_y_right = u_n[nx][ny + 1][nz];
                }

                if (nz == 0) {
                    if (split_type == 0) {
                        u_n_z_left = u_n_z_left_recv[nx * Y + ny];
                    } else {
                        u_n_z_left = u_n[nx][ny][Z - 2];
                    }
                } else {
                    u_n_z_left = u_n[nx][ny][nz - 1];
                }

                if (nz == Z - 1) {
                    if (split_type == 0) {
                        u_n_z_right = u_n_z_right_recv[nx * Y + ny];
                    } else {
                        u_n_z_right = u_n[nx][ny][1];
                    }
                } else {
                    u_n_z_right = u_n[nx][ny][nz + 1];
                }

                u_n_1[nx][ny][nz] =
                    u_n[nx][ny][nz] + dt * dt / 2 *
                                          nabla(u_n[nx][ny][nz], u_n_x_left, u_n_x_right,
                                                u_n_y_left, u_n_y_right, u_n_z_left, u_n_z_right);
            }
        }
    }
    // send border u_1 values
    sendborder(u_n_1, u_n_x_left_send, u_n_y_left_send, u_n_z_left_send, u_n_x_left_recv,
               u_n_y_left_recv, u_n_z_left_recv, u_n_x_right_send, u_n_y_right_send,
               u_n_z_right_send, u_n_x_right_recv, u_n_y_right_recv, u_n_z_right_recv);
}

double main_step(double ***&u_n, double ***&u_n_1, double ***&u_n_2, double *u_n_x_left_send,
                 double *u_n_y_left_send, double *u_n_z_left_send, double *u_n_x_left_recv,
                 double *u_n_y_left_recv, double *u_n_z_left_recv, double *u_n_x_right_send,
                 double *u_n_y_right_send, double *u_n_z_right_send, double *u_n_x_right_recv,
                 double *u_n_y_right_recv, double *u_n_z_right_recv, int nsteps, double dt) {
    double delta = 0;
    // compute u_n_2
#pragma omp parallel for
    for (int nx = 0; nx < X; nx++) {
#pragma omp parallel for
        for (int ny = 0; ny < Y; ny++) {
#pragma omp parallel for
            for (int nz = 0; nz < Z; nz++) {
                double x, y, z;
                double u_n_x_left, u_n_y_left, u_n_z_left, u_n_x_right, u_n_y_right, u_n_z_right;
                x = nx_low * dx + nx * dx;
                y = ny_low * dy + ny * dy;
                z = nz_low * dz + nz * dz;

                if (nx == 0) {
                    if (bnumx == 0) {
                        u_n_2[nx][ny][nz] = 0;
                        continue;
                    } else {
                        u_n_x_left = u_n_x_left_recv[ny * Z + nz];
                    }
                } else {
                    u_n_x_left = u_n_1[nx - 1][ny][nz];
                }

                if (nx == X - 1) {
                    if (bnumx == bmax - 1) {
                        u_n_2[nx][ny][nz] = 0;
                        continue;
                    } else {
                        u_n_x_right = u_n_x_right_recv[ny * Z + nz];
                    }
                } else {
                    u_n_x_right = u_n_1[nx + 1][ny][nz];
                }

                if (ny == 0) {
                    if (split_type == 0) {
                        u_n_y_left = u_n_y_left_recv[nx * Z + nz];
                    } else {
                        u_n_y_left = u_n_1[nx][Y - 2][nz];
                    }
                } else {
                    u_n_y_left = u_n_1[nx][ny - 1][nz];
                }

                if (ny == Y - 1) {
                    if (split_type == 0) {
                        u_n_y_right = u_n_y_right_recv[nx * Z + nz];
                    } else {
                        u_n_y_right = u_n_1[nx][1][nz];
                    }
                } else {
                    u_n_y_right = u_n_1[nx][ny + 1][nz];
                }

                if (nz == 0) {
                    if (split_type == 0) {
                        u_n_z_left = u_n_z_left_recv[nx * Y + ny];
                    } else {
                        u_n_z_left = u_n_1[nx][ny][Z - 2];
                    }
                } else {
                    u_n_z_left = u_n_1[nx][ny][nz - 1];
                }

                if (nz == Z - 1) {
                    if (split_type == 0) {
                        u_n_z_right = u_n_z_right_recv[nx * Y + ny];
                    } else {
                        u_n_z_right = u_n_1[nx][ny][1];
                    }
                } else {
                    u_n_z_right = u_n_1[nx][ny][nz + 1];
                }

                u_n_2[nx][ny][nz] = 2 * u_n_1[nx][ny][nz] - u_n[nx][ny][nz] +
                                    dt * dt *
                                        nabla(u_n_1[nx][ny][nz], u_n_x_left, u_n_x_right,
                                              u_n_y_left, u_n_y_right, u_n_z_left, u_n_z_right);
                double u_n_2_true = u(x, y, z, nsteps * dt);
                delta = abs(u_n_2_true - u_n_2[nx][ny][nz]) > delta
                            ? abs(u_n_2_true - u_n_2[nx][ny][nz])
                            : delta;
            }
        }
    }

    // send border u_n_2 values
    sendborder(u_n_2, u_n_x_left_send, u_n_y_left_send, u_n_z_left_send, u_n_x_left_recv,
               u_n_y_left_recv, u_n_z_left_recv, u_n_x_right_send, u_n_y_right_send,
               u_n_z_right_send, u_n_x_right_recv, u_n_y_right_recv, u_n_z_right_recv);

    // update buffers
    double ***tmp = u_n;
    u_n = u_n_1;
    u_n_1 = u_n_2;
    u_n_2 = tmp;
    return delta;
}

void log(double ***u_n, int N, int num_processes) {
    double *sendbuf = new double[X * Y * Z];
    for (int nx = 0; nx < X; nx++) {
        for (int ny = 0; ny < Y; ny++) {
            for (int nz = 0; nz < Z; nz++) {
                sendbuf[nx * Y * Z + ny * Z + nz] = u_n[nx][ny][nz];
            }
        }
    }
    int range[] = {nx_low, nx_high, ny_low, ny_high, nz_low, nz_high};

    if (get_rank(bnumx, bnumy, bnumz) == 0) {
        fstream fs;
        fs.open("log.txt", fstream::out);

        double *u_computed = new double[N * N * N];
        double *recvbuf = new double[N * N * N];
        int *counts = new int[num_processes];
        int *displs = new int[num_processes];
        int *ranges = new int[num_processes * 6];

        MPI_Gather(range, 6, MPI_INT, ranges, 6, MPI_INT, 0, MPI_COMM_WORLD);

        int X, Y, Z, nx_low, ny_low, nz_low;
        for (int rank = 0, shift = 0; rank < num_processes; rank++) {
            X = ranges[rank * 6 + 1] - ranges[rank * 6 + 0];
            Y = ranges[rank * 6 + 3] - ranges[rank * 6 + 2];
            Z = ranges[rank * 6 + 5] - ranges[rank * 6 + 4];
            counts[rank] = X * Y * Z;
            displs[rank] = shift;
            shift += counts[rank];
        }

        MPI_Gatherv(sendbuf, ::X * ::Y * ::Z, MPI_DOUBLE, recvbuf, counts, displs, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        for (int rank = 0; rank < num_processes; rank++) {
            X = ranges[rank * 6 + 1] - ranges[rank * 6 + 0];
            Y = ranges[rank * 6 + 3] - ranges[rank * 6 + 2];
            Z = ranges[rank * 6 + 5] - ranges[rank * 6 + 4];
            nx_low = ranges[rank * 6 + 0];
            ny_low = ranges[rank * 6 + 2];
            nz_low = ranges[rank * 6 + 4];
            for (int nx = 0; nx < X; nx++) {
                for (int ny = 0; ny < Y; ny++) {
                    for (int nz = 0; nz < Z; nz++) {
                        u_computed[(nx_low + nx) * N * N + (ny_low + ny) * N + (nz_low + nz)] =
                            recvbuf[displs[rank] + nx * Y * Z + ny * Z + nz];
                    }
                }
            }
        }

        for (int nx = 0; nx < N; nx++) {
            for (int ny = 0; ny < N; ny++) {
                for (int nz = 0; nz < N; nz++) {
                    fs << u_computed[nx * N * N + ny * N + nz] << endl;
                }
            }
        }

        delete[] u_computed;
        delete[] recvbuf;
        delete[] counts;
        delete[] displs;
        delete[] ranges;
        fs.close();
    } else {
        MPI_Gather(range, 6, MPI_INT, NULL, 6, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(sendbuf, X * Y * Z, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    delete[] sendbuf;
}

int main(int argc, char *argv[]) {
    if (argc != 5 and argc != 6) {
        cerr << "Usage: task3 Lx Ly Lz N [--logging]" << endl;
        return 1;
    }

    int rank, num_processes;
    int N, K = 250, NSTEPS = argc == 6 ? K : 20;
    double delta;

    MPI_Init(&argc, &argv);
    sscanf(argv[1], "%lf", &Lx);
    sscanf(argv[2], "%lf", &Ly);
    sscanf(argv[3], "%lf", &Lz);
    sscanf(argv[4], "%d", &N);

    dt = T / (K - 1);
    dx = Lx / N;
    dy = Ly / N;
    dz = Lz / N;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    setup_range(rank, N + 1, num_processes);

    double ***u_n_2 = allocate3darray(X, Y, Z);
    double ***u_n_1 = allocate3darray(X, Y, Z);
    double ***u_n = allocate3darray(X, Y, Z);
    double *u_n_x_left_send = new double[Y * Z];
    double *u_n_y_left_send = new double[X * Z];
    double *u_n_z_left_send = new double[X * Y];
    double *u_n_x_left_recv = new double[Y * Z];
    double *u_n_y_left_recv = new double[X * Z];
    double *u_n_z_left_recv = new double[X * Y];
    double *u_n_x_right_send = new double[Y * Z];
    double *u_n_y_right_send = new double[X * Z];
    double *u_n_z_right_send = new double[X * Y];
    double *u_n_x_right_recv = new double[Y * Z];
    double *u_n_y_right_recv = new double[X * Z];
    double *u_n_z_right_recv = new double[X * Y];

    double start_time = MPI_Wtime();

    initialize(u_n, u_n_1, u_n_x_left_send, u_n_y_left_send, u_n_z_left_send, u_n_x_left_recv,
               u_n_y_left_recv, u_n_z_left_recv, u_n_x_right_send, u_n_y_right_send,
               u_n_z_right_send, u_n_x_right_recv, u_n_y_right_recv, u_n_z_right_recv, dt);
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &delta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&delta, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    for (int nsteps = 2; nsteps < NSTEPS; nsteps++) {
        delta = main_step(u_n, u_n_1, u_n_2, u_n_x_left_send, u_n_y_left_send, u_n_z_left_send,
                          u_n_x_left_recv, u_n_y_left_recv, u_n_z_left_recv, u_n_x_right_send,
                          u_n_y_right_send, u_n_z_right_send, u_n_x_right_recv, u_n_y_right_recv,
                          u_n_z_right_recv, nsteps, dt);
        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &delta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&delta, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        cout << "delta " << delta << endl;
        cout << "time " << MPI_Wtime() - start_time << endl;
    }

    if (argc == 6) {
        log(u_n_1, N + 1, num_processes);
    }

    deallocate3darray(u_n_2, X, Y, Z);
    deallocate3darray(u_n_1, X, Y, Z);
    deallocate3darray(u_n, X, Y, Z);
    delete[] u_n_x_left_send;
    delete[] u_n_y_left_send;
    delete[] u_n_z_left_send;
    delete[] u_n_x_left_recv;
    delete[] u_n_y_left_recv;
    delete[] u_n_z_left_recv;
    delete[] u_n_x_right_send;
    delete[] u_n_y_right_send;
    delete[] u_n_z_right_send;
    delete[] u_n_x_right_recv;
    delete[] u_n_y_right_recv;
    delete[] u_n_z_right_recv;

    MPI_Finalize();
    return 0;
}