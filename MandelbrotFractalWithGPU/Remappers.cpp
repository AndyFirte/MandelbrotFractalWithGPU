#include "Remappers.h"
#include <cmath>

double exponentialRemap(
    double x, double x_final,
    double y_initial, double y_final, double y_limit
)
{
    y_initial = y_initial - y_limit;
    y_final = y_final - y_limit;
    return y_initial * pow(y_final / y_initial, x / x_final) + y_limit;
}

double linearRemap
(
    double x,
    double x_initial, double x_final,
    double y_initial, double y_final
)
{
    double m = (y_final - y_initial) / (x_final / x_initial);
    return m * (x - x_initial) + y_initial;
}