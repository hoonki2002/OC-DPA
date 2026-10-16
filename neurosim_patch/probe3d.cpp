// Component-level PPA probe for the heterogeneous-3D split.
//
// Same SubArray set-up as probe.cpp, but it reports every peripheral block on
// its own so the H3D cost model can place each one on its tier: the array,
// row drivers and column mux stay on the memory tier; the SAR ADC and the
// shift-add move to the logic tier. Run once at the memory-tier node and once
// at the logic-tier node and take each block from the node it lives on.
//
// (Original header of probe.cpp follows.)
// Subarray-level PPA probe for the DPA-MoE bit-plane mapping study.
//
// The stock main.cpp runs a whole CNN chip flow and needs per-layer weight and
// activation dumps. We only need the cost of one subarray under a given
// (cell type, cellBit, ADC type, ADC resolution, column sharing), so this
// driver sets up the same globals ProcessingUnit does and reports the ADC /
// accumulation / other breakdown that SubArray already exposes.
//
// Configuration comes from the environment so a sweep needs no recompile.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

static double envd(const char *k, double d) { const char *v = getenv(k); return v ? atof(v) : d; }
static int    envi(const char *k, int d)    { const char *v = getenv(k); return v ? atoi(v) : d; }

int main() {
    // ---- configuration -----------------------------------------------------
    param->memcelltype   = envi("NS_MEMCELL", 2);     // 1 SRAM, 2 RRAM, 3 FeFET
    param->accesstype    = envi("NS_ACCESS", 1);      // 1 CMOS 1T1R, 4 crossbar
    param->cellBit       = envi("NS_CELLBIT", 1);
    param->synapseBit    = envi("NS_SYNAPSEBIT", 8);
    param->numBitInput   = envi("NS_INPUTBIT", 1);
    param->levelOutput   = envi("NS_LEVELOUTPUT", 32);
    param->numColMuxed   = envi("NS_COLMUXED", 8);
    param->SARADC        = envi("NS_SARADC", 0) != 0;
    param->numRowSubArray = envi("NS_ROWS", 128);
    param->numColSubArray = envi("NS_COLS", 128);
    param->technode      = envi("NS_TECHNODE", 22);
    param->resistanceOn  = envd("NS_RON", 6e3);
    param->resistanceOff = param->resistanceOn * envd("NS_ONOFF", 150.0);
    double activityRowRead = envd("NS_ACTIVITY", 1.0);
    // Capacitance of the TSV bus a mux output drives on its way to the ADC on
    // the logic tier (F). Added to the column the way NeuroSim models the
    // bitline: into capCol for the settling time, and into the C*Vread^2
    // bitline term for the read energy.
    double extraColCap = envd("NS_EXTRA_COLCAP_FF", 0.0) * 1e-15;

    param->operationmode = 2;                          // conventionalParallel
    param->conventionalParallel = 1; param->conventionalSequential = 0;
    param->BNNparallelMode = param->BNNsequentialMode = 0;
    param->XNORparallelMode = param->XNORsequentialMode = 0;
    param->parallelRead = 1;
    param->numRowPerSynapse = 1;
    param->numColPerSynapse = (int)ceil((double)param->synapseBit / (double)param->cellBit);
    param->maxConductance = 1.0 / param->resistanceOn;
    param->minConductance = 1.0 / param->resistanceOff;
    param->resistanceAccess = param->resistanceOn * IR_DROP_TOLERANCE;
    if (param->memcelltype == 1) { param->cellBit = 1; param->numColPerSynapse = param->synapseBit; }

    // wire model, same table as Param.cpp
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

    // ---- globals, mirroring ProcessingUnit::ProcessingUnitInitialize --------
    inputParameter.transistorType = conventional;
    inputParameter.deviceRoadmap  = LSTP;
    inputParameter.temperature    = 300;
    inputParameter.processNode    = param->technode;
    tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

    cell.memCellType = (param->memcelltype == 1) ? Type::SRAM
                     : (param->memcelltype == 2) ? Type::RRAM : Type::FeFET;
    cell.accessType  = (param->accesstype == 1) ? CMOS_access
                     : (param->accesstype == 2) ? BJT_access
                     : (param->accesstype == 3) ? diode_access : none_access;
    cell.resistanceOn = param->resistanceOn;
    cell.resistanceOff = param->resistanceOff;
    cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff) / 2;
    cell.readVoltage = param->readVoltage;
    cell.readPulseWidth = param->readPulseWidth;
    cell.accessVoltage = param->accessVoltage;
    cell.resistanceAccess = param->resistanceAccess;
    cell.featureSize = param->featuresize;
    cell.writeVoltage = param->writeVoltage;
    if (cell.memCellType == Type::SRAM) {
        cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;
        cell.widthInFeatureSize  = param->widthInFeatureSizeSRAM;
        cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
        cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
        cell.widthAccessCMOS = param->widthAccessCMOS;
        cell.minSenseVoltage = param->minSenseVoltage;
    } else {
        cell.heightInFeatureSize = (cell.accessType == CMOS_access) ? param->heightInFeatureSize1T1R
                                                                    : param->heightInFeatureSizeCrossbar;
        cell.widthInFeatureSize  = (cell.accessType == CMOS_access) ? param->widthInFeatureSize1T1R
                                                                    : param->widthInFeatureSizeCrossbar;
    }
    double wireLengthRow = param->wireWidth * 1e-9 * cell.heightInFeatureSize;
    double wireLengthCol = param->wireWidth * 1e-9 * cell.widthInFeatureSize;
    Rho *= (1 + 0.00451 * fabs(param->temp - 300));
    param->unitLengthWireResistance = Rho / (param->wireWidth * 1e-9 * param->wireWidth * 1e-9 * AR);
    param->wireResistanceRow = wireLengthRow * param->unitLengthWireResistance;
    param->wireResistanceCol = wireLengthCol * param->unitLengthWireResistance;

    SubArray *sub = new SubArray(inputParameter, tech, cell);
    sub->conventionalParallel = 1; sub->conventionalSequential = 0;
    sub->BNNparallelMode = sub->BNNsequentialMode = 0;
    sub->XNORparallelMode = sub->XNORsequentialMode = 0;
    sub->numRow = param->numRowSubArray; sub->numCol = param->numColSubArray;
    sub->levelOutput = param->levelOutput;
    sub->numColMuxed = param->numColMuxed;
    sub->clkFreq = param->clkFreq;
    sub->relaxArrayCellHeight = param->relaxArrayCellHeight;
    sub->relaxArrayCellWidth = param->relaxArrayCellWidth;
    sub->numReadPulse = param->numBitInput;
    sub->avgWeightBit = param->cellBit;
    sub->numCellPerSynapse = param->numColPerSynapse;
    sub->SARADC = param->SARADC; sub->currentMode = param->currentMode;
    sub->validated = param->validated; sub->spikingMode = NONSPIKING;
    sub->numReadCellPerOperationFPGA = sub->numWriteCellPerOperationFPGA = param->numColSubArray;
    sub->numReadCellPerOperationMemory = param->numColSubArray;
    sub->numWriteCellPerOperationMemory = param->numColSubArray / 8;
    sub->numReadCellPerOperationNeuro = sub->numWriteCellPerOperationNeuro = param->numColSubArray;
    sub->activityRowRead = activityRowRead;

    sub->Initialize(param->numRowSubArray, param->numColSubArray, param->unitLengthWireResistance);
    sub->capCol += extraColCap;
    sub->CalculateArea();

    // Column conductance for an average-case read: half the activated cells on.
    vector<double> columnResistance;
    int rows = param->numRowSubArray;
    for (int j = 0; j < param->numColSubArray; j++) {
        double columnG = 0;
        for (int i = 0; i < rows; i++) {
            double g = (i % 2 == 0) ? param->maxConductance : param->minConductance;
            double r = 1.0 / g + (j + 1) * param->wireResistanceRow
                     + (rows - i) * param->wireResistanceCol
                     + ((cell.accessType == CMOS_access) ? cell.resistanceAccess : 0.0);
            if (i < (int)(rows * activityRowRead)) columnG += 1.0 / r;
        }
        columnResistance.push_back(columnG > 0 ? 1.0 / columnG : 1e20);
    }
    vector<int> partition3D(4, 1);
    // NeuroSim's two-pass protocol: the first pass reports the sensing period
    // that sets the clock, the second reports latency in cycles of that clock.
    sub->CalculateLatency(1e20, columnResistance, true, partition3D);
    double clkPeriod = sub->readLatency;
    if (param->clkFreq > 1.0 / clkPeriod) param->clkFreq = 1.0 / clkPeriod;
    sub->clkFreq = param->clkFreq;
    sub->Initialize(param->numRowSubArray, param->numColSubArray, param->unitLengthWireResistance);
    sub->capCol += extraColCap;
    sub->CalculateArea();
    sub->CalculateLatency(1e20, columnResistance, false, partition3D);
    sub->CalculatePower(columnResistance);
    // the same bitline term NeuroSim uses (capBL*Vread^2 per read cell, times
    // numColMuxed), for the extra capacitance
    double extraE = extraColCap * cell.readVoltage * cell.readVoltage * param->numColSubArray;
    sub->readDynamicEnergyArray += extraE;
    sub->readDynamicEnergy += extraE;
    double T = clkPeriod;

    printf("cfg,memcell,%d,access,%d,cellBit,%d,colPerSyn,%d,levelOutput,%d,colMuxed,%d,sarADC,%d,rows,%d,Ron,%.0f\n",
           param->memcelltype, param->accesstype, param->cellBit, param->numColPerSynapse,
           param->levelOutput, param->numColMuxed, (int)param->SARADC, param->numRowSubArray, param->resistanceOn);
    printf("area_um2,%.4f,areaADC_um2,%.4f,areaAccum_um2,%.4f,areaOther_um2,%.4f,areaArray_um2,%.4f\n",
           sub->usedArea * 1e12, sub->areaADC * 1e12, sub->areaAccum * 1e12,
           sub->areaOther * 1e12, sub->areaArray * 1e12);
    printf("clkPeriod_ns,%.4f,clkFreq_MHz,%.2f\n", T * 1e9, param->clkFreq / 1e6);
    printf("readLatency_ns,%.4f,latADC_ns,%.4f,latAccum_ns,%.4f,latOther_ns,%.4f\n",
           sub->readLatency * T * 1e9, sub->readLatencyADC * T * 1e9,
           sub->readLatencyAccum * T * 1e9, sub->readLatencyOther * T * 1e9);
    printf("readEnergy_pJ,%.6f,eADC_pJ,%.6f,eAccum_pJ,%.6f,eOther_pJ,%.6f,eArray_pJ,%.6f\n",
           sub->readDynamicEnergy * 1e12, sub->readDynamicEnergyADC * 1e12,
           sub->readDynamicEnergyAccum * 1e12, sub->readDynamicEnergyOther * 1e12,
           sub->readDynamicEnergyArray * 1e12);
    #define A(x) (x.area*1e12)
    #define E(x) (x.readDynamicEnergy*1e12)
    printf("comp_area_um2,array,%.4f,wlSwitch,%.4f,slSwitch,%.4f,mux,%.4f,muxDecoder,%.4f,levelShift,%.4f,sarADC,%.4f,shiftAddInput,%.4f,shiftAddWeight,%.4f,height_um,%.4f,width_um,%.4f\n",
           sub->areaArray*1e12, A(sub->wlNewSwitchMatrix)+A(sub->wlSwitchMatrix), A(sub->slSwitchMatrix),
           (param->numColMuxed>1? A(sub->mux):0.0), (param->numColMuxed>1? A(sub->muxDecoder):0.0),
           A(sub->wllevelshifter)+A(sub->bllevelshifter)+A(sub->sllevelshifter), A(sub->sarADC),
           A(sub->shiftAddInput), A(sub->shiftAddWeight), sub->height*1e6, sub->width*1e6);
    printf("comp_energy_pJ,array,%.6f,wlSwitch,%.6f,slSwitch,%.6f,mux,%.6f,muxDecoder,%.6f,sarADC,%.6f,shiftAddInput,%.6f,shiftAddWeight,%.6f\n",
           sub->readDynamicEnergyArray*1e12, E(sub->wlNewSwitchMatrix)+E(sub->wlSwitchMatrix), E(sub->slSwitchMatrix),
           (param->numColMuxed>1? E(sub->mux):0.0), (param->numColMuxed>1? E(sub->muxDecoder):0.0),
           E(sub->sarADC), E(sub->shiftAddInput), E(sub->shiftAddWeight));
    printf("extra_colcap_fF,%.2f,extra_energy_pJ,%.6f\n", extraColCap * 1e15, extraE * 1e12);
    printf("comp_latency_ns,sarADC,%.4f,shiftAddInput,%.4f,colDelay,%.4f\n",
           sub->sarADC.readLatency*1e9, sub->shiftAddInput.readLatency*1e9, sub->colDelay*1e9);
    return 0;
}
