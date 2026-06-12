# PSI-BOIL Phase Change Model → Kynema-SGF

## Project Goal
Transplant the sharp-interface phase change model from PSI-BOIL
(Sato & Niceno 2013, Int. J. Heat Mass Transfer) into Kynema-SGF
to enable local grid refinement (AMR) for boiling simulations.

## Developer
- Yohei Sato, Paul Scherrer Institut (PSI), Laboratory for Scientific
  Multiscale, https://www.psi.ch/en/lsm/people/yohei-sato

## Repository Setup
- Local:    /data/sato/Development/kynema-sgf
- Origin:   https://github.com/yohei-sato-psi/psi-boil-kynema
- Upstream: https://github.com/kynema/kynema-sgf
- Branch:   feature/phase-change-model

## Variable Arrangement Decision
Using the existing Kynema-SGF semi-staggered (MAC) arrangement:
- Velocity u,v,w:      cell centers (primary)
- MAC velocity U^MAC:  cell faces (during projection/advection)
- Pressure p:          nodes (i+1/2, j+1/2, k+1/2)
- Temperature T:       cell centers
- VOF alpha:           cell centers
- rho, mu, lambda:     cell centers
- Mass flux mdot:      cell centers
- Interface normal n:  cell centers (3 components)
- DivU source S_pc:    cell centers

## Phase Change Model (Sato & Niceno 2013)
Key equations:
  mdot = -lambda * (grad_T . n_hat) / h_lv    [mass flux at interface]
  S_pc = mdot * (1/rho_v - 1/rho_l)           [divergence source]
  [u] . n_hat = mdot * (1/rho_v - 1/rho_l)   [velocity jump condition]
  T = T_sat at interface                        [energy BC]

## Key Injection Points in Kynema-SGF Source
1. DivU source term:     src/equation_systems/icns/
                         → add S_pc before ApplyProjection
2. VOF source term:      src/equation_systems/vof/
                         → add mdot source to VOF advection
3. Temperature BC:       src/equation_systems/temperature/
                         → enforce T=T_sat at interface
4. Velocity jump:        U^MAC correction after MAC projection

## New Files to Create
  src/equation_systems/phase_change/
    PhaseChange.H         ← class declaration
    PhaseChange.cpp       ← computes mdot from grad_T at interface
    PhaseChangeSource.H   ← source term registration
    PhaseChangeSource.cpp ← injects source into VOF, momentum, energy

## AMR-Specific Considerations
- Use FillPatchTwoLevels for T ghost cells before grad_T stencil
- Apply velocity jump on fine level before reflux
- Inject S_pc into DivU on each AMR level separately
- Interface normals computed on fine level only near coarse-fine boundary

## Verification Cases (in order)
1. 1D Stefan problem          → tests grad_T extraction and mdot
2. Sucking interface problem   → tests velocity jump condition
3. Bubble growth (Jakob theory)→ full 3D validation
4. Nucleate boiling            → ultimate target

## Current Status (2026-06-12)
- [x] Fresh clone of Kynema-SGF
- [x] Branch feature/phase-change-model created and pushed
- [x] Claude Code installed and configured
- [x] Zalesak disk test case running successfully (translating velocity)
- [ ] Explore src/equation_systems/ to locate DivU injection point
- [ ] Implement PhaseChange module
- [ ] Verify with 1D Stefan problem

## Tools
- Claude Code (terminal): /data/sato/Development/kynema-sgf
- Claude.ai (web): for physics discussion and image analysis
- VisIt: for visualization of simulation results
- Intel oneAPI MPI + MKL available on this machine
- CUDA 12.2 available (NVIDIA GPU support)
