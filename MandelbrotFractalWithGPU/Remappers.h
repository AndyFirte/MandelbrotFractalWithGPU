#ifndef REMAPPERS_H
#define REMAPPERS_H

double exponentialRemap(
    double x, double x_final,
    double y_initial, double y_final, double y_limit
);

double linearRemap
(
    double x,
    double x_initial, double x_final,
    double y_initial, double y_final
);

#endif