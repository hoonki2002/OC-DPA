"""OC-DPA: on-chip dynamic precision allocation for ADC-shared analog in-memory MoE accelerators.

What the code does
------------------
An MoE expert matrix is stored on an analog array as binary planes with 8-bit scales
(`planes.py`). Reading only the first k planes gives a valid k-bit model, so the number
of planes read per (token, expert) is a run-time knob (`analog.py`); fewer planes means
fewer ADC conversions and a shorter time holding a shared converter. An allocator
(`allocators.py`) spends a per-token plane budget on the experts the router weights
most. Two models answer the two questions of the paper:

* quality   `analog.AnalogExperts` replaces a model's expert block and runs the planes
            on a simulated array (8-bit inputs applied bit-serially, one 8-bit ADC conversion
            per input bit, IR-drop, cell variation, 8-bit scale codes): KL divergence and
            downstream accuracy (scripts/21-23).
* hardware  `hw/` and `compare.py` price the same plane counts on a 16-tier 3D stack
            whose ADCs sit on a shared logic tier: area, energy and latency from
            NeuroSim, a closed-form NPU model, and a lane-queue model of prefill
            (scripts/10-12).

A fixed-precision (static) design is evaluated the same way: `analog.AnalogExperts` can hold a
second, separately quantised copy of the experts (`native_bank`, scripts/13), read in place of
the any-precision planes when the plan asks for it.

How the pieces connect (see README.md for the commands)
--------------------------------------------------------
  checkpoint --encode--> planes + scales ----+--> AnalogExperts --> KL / accuracy
        |                (scripts/01)        |    (scripts/21-23)
        +--record routing--> traces ---------+--> hardware model  --> TTFT / area / energy
        |                (scripts/06)        |    (scripts/10-12)
        +--residual table R[l,e,k]-----------+
                         (scripts/03)    allocator input

Modules
-------
  planes.py       stored format: sign planes and 8-bit logarithmic scales
  analog.py       the simulated array: converter, noise model (IR-drop, cell variation),
                  matrix-vector product over the first k planes, expert block
  analog_dense.py non-expert layers on the array (the sensitivity experiment)
  allocators.py   uniform, DPA's dynamic program, the greedy rule and its fixed-point version
  models.py       skeleton loading (a MoE model without its expert tensors), raw layer reads
  fp_experts.py   full-precision reference experts
  encoder.py      the any-precision encoder: nested sign planes with scales (adapted from AnyBCQ)
  irdrop/         wordline/bitline network solver that calibrates the IR-drop gains
  hw/stack.py     3D stack: tiers, TSV bus, ADC lanes, area / energy / read time
  hw/neurosim.py, hw/allocator_ppa.py   wrappers of the NeuroSim driver programs
  hw/npu.py, hw/workload.py             NPU model (GEMM cycles, DRAM traffic) and non-expert work
  system.py       plane plans for a routing trace; decode cost of NPU+flash vs AIMC
  compare.py      prefill / decode / footprint of one design (placement, lane queues)
"""
