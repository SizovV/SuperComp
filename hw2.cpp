#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstdio>

using namespace std;


const double I = M_PI*(exp(1)-2)/2.;
const double V = 4.0;


double func(double x, double y, double z) {
    double var = exp(x * x + y * y) * z;
    return var;
}

double F(double x, double y, double z) {
        if (x >= -1 and x <= 1 and y >= -1 and y <= 1 and z >= 0 and z <= 1 and x*x + y*y + z*z <=1)
                return func(x, y, z);
        else
                return 0;
}

int main(int argc, char *argv[]) {

   if (argc != 2) {
        cerr << "No epsilon" << endl;
        return 1;
    }

    int const portion_size_points = 1000, portion_size_elements = 3 * portion_size_points;

    int processes, id;
    MPI_Status status;

    double start_time, end_time;
    int done = false;

    double integr = 0.0;
    double integral, mc_integral, eps;
    int iterations = 0;

    double *coords;
    double final_mean;

    eps = 0.;
    sscanf(argv[1], "%lf", &eps);

    mc_integral = 0;

    integral = I;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    start_time = MPI_Wtime();

    coords = new double[portion_size_elements];


    while (!done) {
        if (!id) {
            for (int rank = 1; rank < processes; rank++) {
                for (int i = 0; i < portion_size_elements; i++) {
                    coords[i] = (double) rand()/RAND_MAX;
                }
                MPI_Send(coords, portion_size_elements, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);

            }
            iterations += 1;
        } else {
            MPI_Recv(coords, portion_size_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

            integr = 0;

            for (int i = 0; i < portion_size_points; ++i) {
                integr = integr*(i/ (float)(i+1)) + F(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2])/(i+1);
            }
            integr = integr / (processes-1);
        }

        MPI_Reduce(&integr, &mc_integral, 1, MPI_DOUBLE, MPI_SUM, 0 , MPI_COMM_WORLD);

        if (!id) {
            final_mean = final_mean*((iterations - 1)/(float)iterations) + mc_integral/iterations;
            done = fabs(integral - V*final_mean) < eps;
        }

        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    end_time = MPI_Wtime();
    if (!id) {
        cout << "Calculated int: " << V*final_mean << endl;
        cout << "Difference: "  << fabs(integral - V*final_mean) << endl;
        cout << "Points: " << portion_size_elements * iterations << endl;
        cout << "Time:"  << end_time - start_time << endl;

    }

    delete coords;

    MPI_Finalize();
    return 0;
}


