// NeuroSim's own TSV model (TSVPath, Heterogeneous3D) at the logic-tier node.
//
// The H3D cost model moves the ADCs to the logic tier, so digital input bits
// and analog column outputs both cross TSVs. This driver prices one TSV
// crossing exactly as NeuroSim's chip flow does -- driver and receiver
// inverters plus param->tsvCap, energy per bit and latency for a crossing of
// numTSV tiers -- so the stack model takes those numbers instead of a
// hand-written C*V^2.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "TSVPath.h"
#include "Definition.h"
using namespace std;
static int envi(const char *k, int d){const char*v=getenv(k);return v?atoi(v):d;}

int main() {
    int node = envi("TSV_TECHNODE", 7);
    param->numRowSubArray = envi("TSV_ROWS", 128);
    param->numColSubArray = envi("TSV_COLS", 128);
    param->clkFreq = 1e9; param->synchronous = 0;
    inputParameter.transistorType = conventional;
    inputParameter.deviceRoadmap = LSTP;
    inputParameter.temperature = 300;
    inputParameter.processNode = node;
    tech.Initialize(node, LSTP, conventional);
    TSVPath tsv(inputParameter, tech, cell);
    tsv.Initialize();
    tsv.CalculateArea(0, 0, NONE);
    printf("tech,%d,vdd,%.3f,tsvPitch_um,%.3f,tsvCap_fF,%.2f,tsvRes_ohm,%.3f,delta,%.3f,validated,%d\n",
           node, tech.vdd, param->tsvPitch * 1e6, param->tsvCap * 1e15, param->tsvRes, param->delta,
           (int)param->validated);
    printf("area_per_subarray_um2,%.4f,area_per_tsv_um2,%.4f,capInvIn_fF,%.4f,capInvOut_fF,%.4f\n",
           tsv.area * 1e12, tsv.area * 1e12 / (param->numRowSubArray + param->numColSubArray),
           tsv.capInvInput * 1e15, tsv.capInvOutput * 1e15);
    for (int n : {1, 2, 4, 8, 16}) {
        tsv.CalculateLatency(n, 1);
        tsv.CalculatePower(n, 1);
        printf("numTSV,%d,latency_ns,%.5f,energy_fJ_per_bit,%.5f\n",
               n, tsv.readLatency * 1e9, tsv.readDynamicEnergy * 1e15);
    }
    return 0;
}
