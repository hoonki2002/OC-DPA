// Post-ADC recombination: what each mapping costs after the converter.
//
// Conventional bit slicing recombines its slices with the positional weight
// cell_range^s, which is a shift, and applies one per-group scale at the end:
//   one shift-add per slice, then one multiply per (row, group, token).
//
// The plane mapping recombines with a learned per-group alpha_p, which is a
// different constant for every plane, so the multiply cannot be shared:
//   one multiply per (row, group, plane, token).
//
// Both happen after the input bits have been shift-added together, so they run
// once per input vector rather than once per input bit -- a factor of
// numBitInput below the conversion count. NeuroSim has no multiplier, so one is
// built the usual way, as an array of adders: an (a x b) multiply is b-1
// additions of a+b bits.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Adder.h"
#include "ShiftAdd.h"
#include "Definition.h"
using namespace std;
static double envd(const char *k, double d){const char*v=getenv(k);return v?atof(v):d;}
static int    envi(const char *k, int d){const char*v=getenv(k);return v?atoi(v):d;}

int main() {
    const int ADCB  = envi("SU_ADCBITS", 8);      // converter output width
    const int SCB   = envi("SU_SCALEBITS", 16);   // alpha / scale word
    const int UNITS = envi("SU_UNITS", 16);       // columns served in parallel
    const int INB   = envi("SU_INPUTBITS", 8);
    param->technode = envi("SU_TECHNODE", 22);
    const double CLK = envd("SU_CLK", 1e9);

    switch (param->technode) {
        case 22: param->wireWidth = 40; break; case 32: param->wireWidth = 56; break;
        case 14: param->wireWidth = 25; break; default: param->wireWidth = 40; break;
    }
    param->featuresize = param->wireWidth * 1e-9;
    param->clkFreq = CLK; param->synchronous = 0;
    inputParameter.transistorType = conventional;
    inputParameter.deviceRoadmap  = LSTP;
    inputParameter.temperature    = 300;
    inputParameter.processNode    = param->technode;
    tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);
    cell.memCellType = Type::SRAM; cell.featureSize = param->featuresize;

    const int PROD = ADCB + SCB;
    const int STAGES = SCB - 1;                   // adder rows in the array multiplier

    // Input-bit recombination: present in both mappings, one shift-add chain
    // per column over numBitInput partial sums.
    ShiftAdd bitRecomb(inputParameter, tech, cell);
    bitRecomb.Initialize(UNITS, ADCB, CLK, NONSPIKING, INB);
    bitRecomb.CalculateArea(0, 0, NONE);
    bitRecomb.CalculateLatency(1, 0);
    bitRecomb.CalculatePower(1);

    // The multiply, as an adder array.
    Adder mult(inputParameter, tech, cell);
    mult.Initialize(PROD, UNITS * STAGES, CLK);
    mult.CalculateArea(0, 0, NONE);
    mult.CalculateLatency(1e20, 0, 1, 0);
    mult.CalculatePower(1, UNITS * STAGES);

    printf("cfg,adcbits,%d,scalebits,%d,units,%d,inputbits,%d,stages,%d\n",
           ADCB, SCB, UNITS, INB, STAGES);
    printf("mult_pJ_per_col,%.6f,mult_area_um2,%.4f,mult_ns,%.4f\n",
           mult.readDynamicEnergy / UNITS * 1e12, mult.area * 1e12, mult.readLatency * 1e9);
    printf("recomb_pJ_per_col,%.6f,recomb_area_um2,%.4f\n",
           bitRecomb.readDynamicEnergy / UNITS * 1e12, bitRecomb.area * 1e12);
    return 0;
}
