// include/data_utils.hpp
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

inline void initialize_data(float* data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

inline bool check_result(
    const float* expected,
    const float* actual,
    int n,
    float tolerance = 1e-5f
)
{
    for (int i = 0; i < n; ++i)
    {
        float difference = std::fabs(expected[i] - actual[i]);

        if (difference > tolerance)
        {
            std::fprintf(
                stderr,
                "Result mismatch at index %d: "
                "expected = %.6f, actual = %.6f, difference = %.6f\n",
                i,
                expected[i],
                actual[i],
                difference
            );

            return false;
        }
    }

    return true;
}