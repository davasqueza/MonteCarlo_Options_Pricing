#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "asa241.h"
#include "omp.h"

double TSLA[] = {
    379.810,
    385.000,
    375.100,
    373.910,
    366.480,
    351.090,
    344.990,
    345.250,
    340.970,
    339.600,
    341.100,
    341.530,
    348.140,
    355.010,
    355.330,
    356.880,
    342.940,
    355.590,
    354.600,
    355.680,
    355.570,
    350.600,
    355.750,
    359.650,
    351.810,
    345.100,
    337.020,
    337.340,
    325.840,
    326.170,
    320.870,
    320.080,
    331.530,
    321.080,
    299.260,
    306.090,
    302.780,
    306.050,
    304.390,
    302.990,
    302.990,
    315.400,
    308.700,
    311.300,
    312.500,
    315.050,
    308.740,
    317.810,
    312.600,
    315.550,
    316.810,
    317.550,
    307.540,
    308.850,
    306.530,
    305.200,
    303.700,
    313.260,
    311.240,
    315.130,
    328.910,
    341.030,
    339.030,
    337.890,
    343.450,
    338.870,
    331.100,
    328.980,
    331.660,
    325.200,
    317.290,
    311.640,
    315.360,
    311.350,
    320.530,
    317.250,
    314.620,
    316.580,
    336.410,
    333.690,
    334.800,
    337.950,
    336.220,
    340.060,
    347.160,
    344.570,
    350.020,
    351.560,
    352.790,
    345.890,
    337.640,
    342.850,
    349.530,
    345.820,
    354.310,
    349.250,
    343.750,
    333.130,
    333.970,
    345.000,
    315.230,
    310.420,
    315.730,
    323.660,
    322.310,
    334.070,
    335.490,
    334.770,
    333.300,
    346.170,
    352.050,
    357.420,
    350.990,
    343.060,
    330.930,
    335.120,
    333.350,
    328.200,
    332.300,
    329.100,
    327.170,
    345.510,
    341.840,
    326.630,
    325.600,
    321.350,
    313.560,
    310.550,
    316.530,
    309.100,
    301.540,
    304.180,
    279.180,
    257.780,
    266.130,
    252.480,
    267.530,
    286.940,
    305.720,
    299.300,
    289.660,
    304.700,
    300.930,
    294.080,
    300.340,
    291.210,
    287.690,
    293.350,
    300.080,
    290.240,
    283.370,
    283.460,
    280.690,
    285.480,
    294.080,
    293.900,
    299.920,
    301.150,
    284.450,
    294.090,
    302.770,
    301.970,
    306.850,
    305.020,
    301.060,
    291.970,
    284.180,
    286.480,
    284.540,
    276.820,
    284.490,
    275.010,
    279.070,
    277.850,
    278.850,
    283.760,
    291.720,
    284.730,
    291.820,
    296.740,
    291.130,
    319.500,
    316.090,
    317.660,
    332.100,
    342.770,
    344.780,
    357.720,
    358.170,
    370.830,
    352.550,
    362.220,
    347.510,
    333.630,
    333.010,
    342.000,
    344.500,
    349.930,
    342.950,
    335.070,
    310.860,
    309.160,
    308.900,
    318.510,
    322.470,
    318.960,
    316.710,
    318.870,
    310.100,
    322.690,
    323.850,
    320.230,
    313.580,
    303.200,
    297.430,
    308.740,
    306.650,
    297.180,
    290.170,
    298.140,
    300.840,
    349.540,
    348.170,
    341.990,
    379.570,
    370.340,
    352.450,
    355.490,
    356.410,
    347.640,
    338.690,
    335.450,
    305.500,
    308.440,
    321.900,
    321.640,
    320.100,
    322.820,
    319.270,
    311.860,
    305.010,
    303.150,
    301.660,
    288.950,
    280.740,
    280.950,
    263.240,
    285.500,
    279.440,
    290.540,
    289.460,
    295.200
};

double monteCarloSim(double previousValue, double stdDev, double drift){
    double random = (double)rand() / (double)RAND_MAX;
    return previousValue * exp(drift + stdDev * r8_normal_01_cdf_inverse(random));
}

void computeStep(int simulationLength, double TSLA_SIM[], int totalValues, double stdDev, double drift){
    //Initialize first value of simulation
    TSLA_SIM[0] = monteCarloSim(TSLA[totalValues - 2], stdDev, drift);
    printf("Simulation [0]: %f\n", TSLA_SIM[0]);

    for(int simValueIndex = 1; simValueIndex < simulationLength; simValueIndex++){
        TSLA_SIM[simValueIndex] = monteCarloSim(TSLA[totalValues - 2], stdDev, drift);
        printf("Simulation [%i]: %f\n", simValueIndex, TSLA_SIM[simValueIndex]);
    }
}

int main() {
    printf("Monte Carlo Simulation, TSLA\n");
    srand((unsigned int) time(NULL));

    int totalValues = sizeof(TSLA) / sizeof(double);
    int simulationLength = 20;
    int maxProcessors = omp_get_num_procs();
    int totalSteps = 100;
    int stepsByProcessor = (int)ceil(totalSteps / maxProcessors);

    double TSLA_PDR[totalValues], TSLA_SIM[simulationLength], average, variance, drift;
    double sum = 0.0;
    double stdDev = 0.0;

    //Compute PDR (Periodic Daily Return)
    for(int closeValueIndex = 0; closeValueIndex < totalValues - 1; closeValueIndex++){
        TSLA_PDR[closeValueIndex] = log(TSLA[closeValueIndex] / TSLA[closeValueIndex + 1]);
    }

    //Compute summation of close values
    for(int closeValueIndex = 0; closeValueIndex < totalValues; closeValueIndex++){
        sum += TSLA_PDR[closeValueIndex];
    }

    //Compute average
    average = sum / (totalValues - 1);

    //Compute standard deviation
    for(int closeValueIndex = 0; closeValueIndex < totalValues; closeValueIndex++){
        stdDev += pow(TSLA_PDR[closeValueIndex] - average, 2);
    }

    stdDev = sqrt(stdDev/(totalValues - 1));

    //Compute variance
    variance = pow(stdDev, 2);

    //Compute drift
    drift = average - (variance / 2);

    printf("Average: %f\n", average);
    printf("Variance: %f\n", variance);
    printf("stdDev: %f\n", stdDev);
    printf("drift: %f\n", drift);
    printf("# processors: %i\n", maxProcessors);

    #pragma omp parallel num_threads(maxProcessors)
    {
        int processorNum  = omp_get_thread_num();
        int minStep = processorNum * stepsByProcessor;
        int maxStep = (processorNum + 1) * stepsByProcessor;

        if(maxStep > totalSteps){
            maxStep = totalSteps;
        }

        printf("Range: [%i,%i]\n", minStep, maxStep);

        for(int step = minStep; step < maxStep; step++){
            printf("--------------------------\n");
            printf("-- Monte Carlo / Step %i --\n", step);
            printf("--------------------------\n");
            computeStep(simulationLength, TSLA_SIM, totalValues, stdDev, drift);
        }
    }

    return 0;
}