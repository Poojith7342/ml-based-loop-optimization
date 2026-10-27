#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef float DT;
#define SCALE 1.0
#define N 65536
static DT x0[N], x1[N], y[N];

// KERNEL_BEGIN
__attribute__((noinline, noipa)) DT kernel(int r)
{
    for (int i = 0; i < N; i++) {
        DT v = x0[i];
        y[i] = ((0.300*v + 0.200)*v + 0.100)*v + 0.500*x0[i] + 0.250*x1[i] + 0.250*y[i];
    }
    return y[0];
}
// KERNEL_END

static unsigned long long rng_state = 88172645463325252ULL;
static double rnd(void){ rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
                         rng_state ^= rng_state << 17;
                         return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                            return (double)t.tv_sec * 1e9 + (double)t.tv_nsec; }
#define FILL(arr) for (long q = 0; q < (long)(sizeof(arr)/sizeof(DT)); q++) ((DT*)arr)[q] = (DT)(SCALE * rnd())
#define SUM(arr)  for (long q = 0; q < (long)(sizeof(arr)/sizeof(DT)); q++) chk += (double)((DT*)arr)[q]
static void init(void){ rng_state = 88172645463325252ULL; FILL(x0); FILL(x1); FILL(y); }
int main(void){
    volatile DT sink = 0;
    init();
    double chk = 0;
    for (int r = 0; r < 3; r++) chk += (double)kernel(r);
    SUM(x0); SUM(x1); SUM(y);
    init();
    sink = sink + kernel(0);
    double t0 = now_ns();
    sink = sink + kernel(1);
    double t1 = now_ns() - t0;
    if (t1 < 1) t1 = 1;
    long reps = (long)(15e6 / t1);
    if (reps < 1) reps = 1;
    if (reps > 2000000) reps = 2000000;
    double best = 1e30;
    for (int trial = 0; trial < 5; trial++) {
        t0 = now_ns();
        for (long k = 0; k < reps; k++) sink = sink + kernel((int)k);
        double dt = (now_ns() - t0) / (double)reps;
        if (dt < best) best = dt;
    }
    printf("%.3f %.9e\n", best, chk);
    return 0;
}
