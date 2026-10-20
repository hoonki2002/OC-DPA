// Precision Allocator: PPA of the per-dispatch plane-budget allocator, built
// from the same NeuroSim primitive library and technology node as the analog
// subarray it sits beside.
//
// What the circuit does
// --------------------
// A dispatch routes n experts. Every expert starts at KMIN planes -- the
// budget B = kbar*n always admits that, since kbar >= KMIN -- and the
// remaining B - n*KMIN plane-grants go one at a time to whichever expert
// currently offers the largest drop in weighted residual. Measurement shows
// this marginal-gain rule reproduces the published dynamic program's
// allocation exactly, because the residual curves are convex in k, so the
// hardware needs a max-select and a counter rather than a DP table.
//
// Arithmetic
// ----------
// The quantity compared is g_i^2 * dR_i(k_i). Only its ordering matters, so
// the datapath carries the natural log of it:
//
//     s_i = 2*z_i + ln dR[L, e_i, k_i]
//
// where z_i is the router logit for expert i. Two consequences, both of which
// remove hardware: the softmax normaliser is a per-dispatch constant and so
// cannot change the ordering, which deletes the softmax and the squarer; and
// the product becomes a sum, which deletes the multiplier. Subtracting the
// top-1 logit -- already found by the top-k selector -- keeps s_i non-positive
// and bounds the word width.
//
// The max-select returns an index, not a value, so the comparison word is
// {s_i, i}: appending log2(n) index bits makes the tournament's winning word
// carry the winning expert in its low bits, and breaks ties deterministically.
//
// Blocks
// ------
//   LUT       SRAM macro holding ln dR, one row per (layer, expert)
//   mux       selects the current k_i's field out of the LUT word
//   adder     s_i = 2*z_i + ln dR, n in parallel
//   argmax    MaxPooling tournament tree over the n comparison words
//   regs      k_i counters, s_i, and the sequencer state
//   incr      increments the winning expert's k
//
// Configuration comes from the environment so a sweep needs no recompile.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Adder.h"
#include "Mux.h"
#include "DFF.h"
#include "MaxPooling.h"
#include "Comparator.h"
#include "RowDecoder.h"
#include "Precharger.h"
#include "SenseAmp.h"
#include "SRAMWriteDriver.h"
#include "Definition.h"

using namespace std;

static double envd(const char *k, double d) { const char *v = getenv(k); return v ? atof(v) : d; }
static int    envi(const char *k, int d)    { const char *v = getenv(k); return v ? atoi(v) : d; }

int main() {
    // ---- configuration -----------------------------------------------------
    const int N        = envi("PA_N", 8);          // routed experts per dispatch
    const int KMIN     = envi("PA_KMIN", 2);
    const int KMAX     = envi("PA_KMAX", 8);
    const int KBAR     = envi("PA_KBAR", 4);
    const int QLUT     = envi("PA_QLUT", 9);       // bits per stored ln-gain
    const int QS       = envi("PA_QS", 12);        // adder / compare width
    const int LUTROWS  = envi("PA_LUTROWS", 1024); // layers * experts
    const int BANKS    = envi("PA_BANKS", 1);      // LUT banks (N -> one-cycle read)
    param->technode    = envi("PA_TECHNODE", 22);
    const double CLK   = envd("PA_CLK", 1e9);
    // 0: marginal-gain allocator (this work).  1: the published dynamic
    // program, built from the same library, as the cost of not having the
    // convexity result.
    const int MODE     = envi("PA_MODE", 0);
    const int DPPAR    = envi("PA_DPPAR", 1);     // transitions evaluated per cycle

    const int KR    = KMAX - KMIN;                 // gain entries per expert
    const int IDXB  = (int)ceil(log2((double)N));
    const int QCMP  = QS + IDXB;                   // {s_i, i}
    const int STEPS = KBAR * N - N * KMIN;         // plane-grants to hand out
    const int LUTCOLS = KR * QLUT;

    // ---- technology, mirroring probe.cpp -----------------------------------
    switch (param->technode) {
        case 130: param->wireWidth = 175; break; case 90: param->wireWidth = 110; break;
        case 65:  param->wireWidth = 105; break; case 45: param->wireWidth = 80;  break;
        case 32:  param->wireWidth = 56;  break; case 22: param->wireWidth = 40;  break;
        case 14:  param->wireWidth = 25;  break; default: param->wireWidth = 18;  break;
    }
    double AR, Rho;
    switch (param->wireWidth) {
        case 175: AR = 1.60; Rho = 2.20e-8; break; case 110: AR = 1.60; Rho = 2.52e-8; break;
        case 105: AR = 1.70; Rho = 2.68e-8; break; case 80:  AR = 1.70; Rho = 3.31e-8; break;
        case 56:  AR = 1.80; Rho = 3.70e-8; break; case 40:  AR = 1.90; Rho = 4.03e-8; break;
        case 25:  AR = 2.00; Rho = 5.08e-8; break; default:  AR = 2.00; Rho = 6.35e-8; break;
    }
    param->featuresize = param->wireWidth * 1e-9;
    param->clkFreq = CLK;
    // Adder, DFF and MaxPooling report cycle counts when param->synchronous is
    // set, which is what a fixed-schedule NeuroSim pipeline wants. This is a
    // custom datapath whose schedule we define, so we take the combinational
    // delays in seconds and derive the clock from them.
    param->synchronous = 0;

    inputParameter.transistorType = conventional;
    inputParameter.deviceRoadmap  = LSTP;
    inputParameter.temperature    = 300;
    inputParameter.processNode    = param->technode;
    tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

    cell.memCellType       = Type::SRAM;
    cell.accessType        = CMOS_access;
    cell.featureSize       = param->featuresize;
    cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;
    cell.widthInFeatureSize  = param->widthInFeatureSizeSRAM;
    cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
    cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
    cell.widthAccessCMOS   = param->widthAccessCMOS;
    cell.minSenseVoltage   = param->minSenseVoltage;

    Rho *= (1 + 0.00451 * fabs(param->temp - 300));
    param->unitLengthWireResistance = Rho / (param->wireWidth * 1e-9 * param->wireWidth * 1e-9 * AR);

    // ---- LUT macro ---------------------------------------------------------
    const int lutRowsPerBank = (int)ceil((double)LUTROWS / BANKS);
    double lutLengthRow = (double)LUTCOLS * cell.widthInFeatureSize * tech.featureSize;
    double lutLengthCol = (double)lutRowsPerBank * cell.heightInFeatureSize * tech.featureSize;
    double capRow = lutLengthRow * 0.2e-15 / 1e-6;
    double capCol = lutLengthCol * 0.2e-15 / 1e-6;
    double resRow = lutLengthRow * param->unitLengthWireResistance;
    double resCol = lutLengthCol * param->unitLengthWireResistance;
    double resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, 300, tech, 0);
    double capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS,
                                             cell.widthInFeatureSize * tech.featureSize, tech);
    cell.capSRAMCell = capCellAccess
        + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech)
        + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech)
        + CalculateGateCap(cell.widthSRAMCellNMOS * tech.featureSize, tech)
        + CalculateGateCap(cell.widthSRAMCellPMOS * tech.featureSize, tech);

    RowDecoder      lutDecoder(inputParameter, tech, cell);
    Precharger      lutPrecharge(inputParameter, tech, cell);
    SenseAmp        lutSense(inputParameter, tech, cell);
    SRAMWriteDriver lutWrite(inputParameter, tech, cell);

    lutDecoder.Initialize(REGULAR_ROW, (int)ceil(log2((double)lutRowsPerBank)), false, false);
    lutPrecharge.Initialize(LUTCOLS, resCol, 1, LUTCOLS, LUTCOLS);
    lutSense.Initialize(LUTCOLS, false, cell.minSenseVoltage, lutLengthRow / LUTCOLS, CLK, LUTCOLS);
    lutWrite.Initialize(LUTCOLS, 1, LUTCOLS);

    lutDecoder.CalculateArea(lutLengthCol, 0, NONE);
    lutPrecharge.CalculateArea(0, lutLengthRow, NONE);
    lutSense.CalculateArea(0, lutLengthRow, NONE);
    lutWrite.CalculateArea(0, lutLengthRow, NONE);
    double lutArrayArea = (double)lutRowsPerBank * LUTCOLS
                        * cell.heightInFeatureSize * cell.widthInFeatureSize
                        * tech.featureSize * tech.featureSize;

    // One dispatch reads the rows of its n routed experts.
    const double lutReads = (double)N / BANKS;
    lutDecoder.CalculateLatency(1e20, capRow, 0, lutReads, 0, 0);
    lutPrecharge.CalculateLatency(1e20, capCol, lutReads, 0, 0);
    lutSense.CalculateLatency(lutReads);
    lutDecoder.CalculatePower(lutReads, 0);
    lutPrecharge.CalculatePower(lutReads, 0);
    lutSense.CalculatePower(lutReads);

    double lutArea = (lutArrayArea + lutDecoder.area + lutPrecharge.area + lutSense.area + lutWrite.area) * BANKS;
    double lutLatency = lutDecoder.readLatency + lutPrecharge.readLatency + lutSense.readLatency;
    double lutEnergy = (lutDecoder.readDynamicEnergy + lutPrecharge.readDynamicEnergy
                        + lutSense.readDynamicEnergy) * BANKS;
    double lutLeak = (lutDecoder.leakage + lutPrecharge.leakage + lutSense.leakage + lutWrite.leakage) * BANKS;

    // ---- iteration datapath ------------------------------------------------
    double dpLeakExtra = 0;
    Mux        lutMux(inputParameter, tech, cell);
    RowDecoder muxDecoder(inputParameter, tech, cell);
    Adder      sAdder(inputParameter, tech, cell);
    MaxPooling argmax(inputParameter, tech, cell);
    DFF        regs(inputParameter, tech, cell);
    Adder      incr(inputParameter, tech, cell);

    // One KR:1 mux per bit of each expert's selected field.
    lutMux.Initialize(N * QLUT, KR, resCellAccess, false);
    muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2((double)KR)), true, false);
    sAdder.Initialize(QS, N, CLK);
    argmax.Initialize(QCMP, N, 1, CLK);
    // k counters, the comparison words, and the sequencer's step counter.
    regs.Initialize(N * (QS + 3) + 8, CLK);
    incr.Initialize(3, 1, CLK);

    lutMux.CalculateArea(0, 0, NONE);
    muxDecoder.CalculateArea(0, 0, NONE);
    sAdder.CalculateArea(0, 0, NONE);
    // MaxPooling reports a unit area that must be computed before the block is
    // laid out; it abuts the LUT, so it inherits the LUT's row pitch.
    argmax.CalculateUnitArea(NONE);
    argmax.CalculateArea(lutLengthRow);
    regs.CalculateArea(0, 0, NONE);
    incr.CalculateArea(0, 0, NONE);

    // One activation each, to read off the combinational delay of one stage.
    const double steps = (double)STEPS;
    lutMux.CalculateLatency(1e20, 0, 1);
    muxDecoder.CalculateLatency(1e20, lutMux.capTgGateN * N, lutMux.capTgGateP * N, 1, 0, 0);
    sAdder.CalculateLatency(1e20, 0, 1, 0);
    argmax.CalculateLatency(1e20, 0, 1, 0);
    incr.CalculateLatency(1e20, 0, 1, 0);
    regs.CalculateLatency(1e20, 1);

    lutMux.CalculatePower(steps);
    muxDecoder.CalculatePower(steps, 0);
    sAdder.CalculatePower(steps, N);
    argmax.CalculatePower(steps);
    // Only the winning expert's counter and comparison word are rewritten on a
    // step; the other registers hold. Charging every flop every cycle would
    // flatter neither variant fairly, so both are charged for written bits.
    regs.CalculatePower(steps, QS + 3, param->validated);
    incr.CalculatePower(steps, 1);

    // ---- dynamic-program variant ------------------------------------------
    // The DP carries a cost table over every reachable partial budget and a
    // backpointer for every (expert, budget) pair, and walks
    // n * (B+1) * krange transitions. Both the state and the walk are what the
    // convexity of the residual curves removes.
    Adder      dpAdd(inputParameter, tech, cell);
    Comparator dpCmp(inputParameter, tech, cell);
    DFF        dpState(inputParameter, tech, cell);
    DFF        dpBack(inputParameter, tech, cell);
    double dpExtraArea = 0, dpExtraEnergy = 0, dpExtraLeak = 0, dpPath = 0;
    double dpCycles = 0;
    const int BUD = KBAR * N;
    if (MODE == 1) {
        const double trans = (double)N * (BUD + 1) * (KMAX - KMIN + 1);
        dpCycles = ceil(trans / DPPAR);
        dpAdd.Initialize(QS, DPPAR, CLK);
        dpCmp.Initialize(QS, DPPAR);
        // Two cost rows, ping-ponged, plus the backpointer array.
        dpState.Initialize(2 * (BUD + 1) * QS, CLK);
        dpBack.Initialize(N * (BUD + 1) * 3, CLK);
        dpAdd.CalculateArea(0, 0, NONE);
        dpCmp.CalculateUnitArea(NONE);
        dpCmp.CalculateArea(lutLengthRow);
        dpState.CalculateArea(0, 0, NONE);
        dpBack.CalculateArea(0, 0, NONE);
        dpAdd.CalculateLatency(1e20, 0, 1, 0);
        dpCmp.CalculateLatency(1e20, 0, 1, 0);
        dpAdd.CalculatePower(dpCycles, DPPAR);
        dpCmp.CalculatePower(dpCycles, DPPAR);
        dpState.CalculatePower(dpCycles, DPPAR * QS, param->validated);
        dpBack.CalculatePower(dpCycles, DPPAR * 3, param->validated);
        dpExtraArea = dpAdd.area + dpCmp.area + dpState.area + dpBack.area;
        dpExtraEnergy = dpAdd.readDynamicEnergy + dpCmp.readDynamicEnergy
                      + dpState.readDynamicEnergy + dpBack.readDynamicEnergy;
        dpExtraLeak = dpAdd.leakage + dpCmp.leakage + dpState.leakage + dpBack.leakage;
        dpPath = dpAdd.readLatency + dpCmp.readLatency;
    }

    double dpArea = lutMux.area + muxDecoder.area + sAdder.area + argmax.area + regs.area + incr.area;
    if (MODE == 1) { dpArea += dpExtraArea; dpLeakExtra = dpExtraLeak; }
    double dpEnergy = lutMux.readDynamicEnergy + muxDecoder.readDynamicEnergy + sAdder.readDynamicEnergy
                    + argmax.readDynamicEnergy + regs.readDynamicEnergy + incr.readDynamicEnergy
                    + dpExtraEnergy;
    double dpLeak = dpLeakExtra + lutMux.leakage + muxDecoder.leakage + sAdder.leakage + argmax.leakage
                  + regs.leakage + incr.leakage;

    // One allocation step: select the current gain, add the gate term, run the
    // tournament. The register delay is NeuroSim's clock-derived DFF proxy, so
    // it is reported separately from the gate-level combinational path.
    double stepPath = (MODE == 1) ? dpPath
                                  : lutMux.readLatency + sAdder.readLatency + argmax.readLatency;
    double lutPath  = (lutDecoder.readLatency + lutPrecharge.readLatency + lutSense.readLatency) / lutReads;
    double critical = max(stepPath, lutPath);
    double fmax = 1.0 / critical;
    double cycles = lutReads + ((MODE == 1) ? dpCycles + N : steps) + 2;
    double dispatchLatency = cycles * critical;

    // ---- report ------------------------------------------------------------
    printf("cfg,mode,%d,N,%d,kbar,%d,steps,%d,QLUT,%d,QS,%d,QCMP,%d,LUTrows,%d,LUTcols,%d,banks,%d,node,%d\n",
           MODE, N, KBAR, STEPS, QLUT, QS, QCMP, LUTROWS, LUTCOLS, BANKS, param->technode);
    printf("area_um2,%.4f,lut_um2,%.4f,lutarray_um2,%.4f,mux_um2,%.4f,adder_um2,%.4f,argmax_um2,%.4f,regs_um2,%.4f\n",
           (lutArea + dpArea) * 1e12, lutArea * 1e12, lutArrayArea * BANKS * 1e12,
           (lutMux.area + muxDecoder.area) * 1e12, (sAdder.area + incr.area) * 1e12,
           argmax.area * 1e12, regs.area * 1e12);
    printf("critical_ns,%.4f,fmax_MHz,%.1f,cycles,%.0f,latency_ns,%.4f\n",
           critical * 1e9, fmax / 1e6, cycles, dispatchLatency * 1e9);
    printf("energy_pJ,%.6f,lut_pJ,%.6f,datapath_pJ,%.6f,leakage_uW,%.4f\n",
           (lutEnergy + dpEnergy) * 1e12, lutEnergy * 1e12, dpEnergy * 1e12,
           (lutLeak + dpLeak) * 1e6);
    printf("lut_bits,%d,lut_KB,%.2f\n", LUTROWS * LUTCOLS, LUTROWS * LUTCOLS / 8192.0);
    if (getenv("PA_DEBUG")) {
        printf("dbg,mux,%.6g,muxdec,%.6g,adder,%.6g,argmax,%.6g,dff,%.6g,incr,%.6g,"
               "lutdec,%.6g,prech,%.6g,sense,%.6g\n",
               lutMux.readLatency, muxDecoder.readLatency, sAdder.readLatency,
               argmax.readLatency, regs.readLatency, incr.readLatency,
               lutDecoder.readLatency, lutPrecharge.readLatency, lutSense.readLatency);
    }
    return 0;
}
