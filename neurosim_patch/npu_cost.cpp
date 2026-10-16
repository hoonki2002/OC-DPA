// Digital NPU costs from NeuroSim's own primitives, at the logic-tier node.
//
// The dense path runs on a weight-stationary systolic array on the 7 nm logic
// tier. Pricing it with the same circuit library as the analog array keeps the
// two halves of the chip on one methodology. NeuroSim has no multiplier, so --
// as in scaleunit.cpp -- an a x b multiply is b-1 additions of a+b bits; each
// PE adds a 32-bit accumulate and its pipeline registers. On-chip buffers are
// NeuroSim SRAM Buffer macros.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <string>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Adder.h"
#include "DFF.h"
#include "Buffer.h"
#include "Definition.h"
using namespace std;
static int envi(const char *k, int d){const char*v=getenv(k);return v?atoi(v):d;}

int main() {
    param->technode = envi("NPU_TECHNODE", 7);
    const int PES = envi("NPU_PES", 128 * 128);
    const int SRAM_KB = envi("NPU_SRAM_KB", 2048);      // one buffer
    const double CLK = 1e9;
    switch (param->technode) {
        case 22: param->wireWidth = 40; break; case 14: param->wireWidth = 25; break;
        default: param->wireWidth = 18; break;
    }
    double AR = (param->wireWidth == 40) ? 1.90 : (param->wireWidth == 25 ? 2.00 : 2.00);
    double Rho = (param->wireWidth == 40) ? 4.03e-8 : (param->wireWidth == 25 ? 5.08e-8 : 6.35e-8);
    param->featuresize = param->wireWidth * 1e-9;
    param->unitLengthWireResistance = Rho / (param->featuresize * param->featuresize * AR);
    param->clkFreq = CLK; param->synchronous = 0;
    inputParameter.transistorType = conventional;
    inputParameter.deviceRoadmap  = LSTP;
    inputParameter.temperature    = 300;
    inputParameter.processNode    = param->technode;
    tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);
    cell.memCellType = Type::SRAM; cell.featureSize = param->featuresize;
    cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;
    cell.widthInFeatureSize = param->widthInFeatureSizeSRAM;
    cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
    cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
    cell.widthAccessCMOS = param->widthAccessCMOS;
    cell.minSenseVoltage = param->minSenseVoltage;

    // one PE: 8x8 multiply (7 adders of 16 b) + 32-b accumulate
    Adder mul(inputParameter, tech, cell);
    mul.Initialize(16, 7, CLK);
    mul.CalculateArea(0, 0, NONE); mul.CalculatePower(1, 7);
    Adder acc(inputParameter, tech, cell);
    acc.Initialize(32, 1, CLK);
    acc.CalculateArea(0, 0, NONE); acc.CalculatePower(1, 1);
    DFF reg(inputParameter, tech, cell);                  // weight 8 + input 8 + psum 32
    reg.Initialize(48, CLK);
    reg.CalculateArea(0, 0, NONE); reg.CalculatePower(1, 48, false);
    double pe_pJ = (mul.readDynamicEnergy + acc.readDynamicEnergy + reg.readDynamicEnergy) * 1e12;
    double pe_um2 = (mul.area + acc.area + reg.area) * 1e12;

    // SRAM buffer: 32 kB banks of 256 rows x 1024 columns; an access touches
    // one bank, the area is the bank times the bank count
    int width = 1024;
    int bank_bits = 256 * width;
    int banks = SRAM_KB * 1024 * 8 / bank_bits;
    Buffer buf(inputParameter, tech, cell);
    buf.Initialize(bank_bits, width, 1, param->unitLengthWireResistance, CLK, true);
    buf.CalculateArea(0, 1e-4, NONE);
    buf.CalculateLatency(width, 1, width, 1, 0);
    buf.CalculatePower(width, 1, width, 1);
    double rd_pJ_per_B = buf.readDynamicEnergy * 1e12 / (width / 8);
    double wr_pJ_per_B = buf.writeDynamicEnergy * 1e12 / (width / 8);

    printf("tech,%d,pe_pJ_per_mac,%.6f,pe_um2,%.4f,array_mm2,%.4f,mul_pJ,%.6f,acc_pJ,%.6f,reg_pJ,%.6f\n",
           param->technode, pe_pJ, pe_um2, pe_um2 * PES / 1e6,
           mul.readDynamicEnergy * 1e12, acc.readDynamicEnergy * 1e12, reg.readDynamicEnergy * 1e12);
    printf("sram_kb,%d,sram_mm2,%.4f,sram_rd_pJ_per_B,%.6f,sram_wr_pJ_per_B,%.6f\n",
           SRAM_KB, buf.area * banks * 1e6, rd_pJ_per_B, wr_pJ_per_B);
    return 0;
}
