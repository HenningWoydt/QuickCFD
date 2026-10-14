# Todo

## Runtime-selectable velocity field

Currently the velocity field (deformational / rotating / vortex) is compiled into the binary via `face_velocity.H`. Switching simulations requires modifying the header and recompiling, which means all three `amr/inputs/` files share one binary that only implements one flow at a time.

**Goal:** add `adv.velocity_type = deformational|rotating|vortex` to the inputs file so all three simulations can be run without recompiling.

**Plan:**

1. Add `int velocity_type` member to `AmrCoreAdv` (in `Source/AmrCoreAdv.H`)
2. Read it in `ReadParameters()` via `ParmParse` (`adv.velocity_type`, default `0`)
3. Pass it into `DefineVelocityAtLevel` and capture by value in the GPU lambda
4. In `face_velocity.H`, replace `get_face_velocity_psi` with three named functions:
   - `get_face_velocity_psi_deformational` — original LeVeque sin²·sin²·cos(πt/2)/π
   - `get_face_velocity_psi_rotating`      — solid-body: ψ = 0.5((x−0.5)²+(y−0.5)²)
   - `get_face_velocity_psi_vortex`        — regularised point vortex: ψ = (Γ/4π)ln(r²+ε²)
5. Dispatch on `velocity_type` inside the `ParallelFor` lambda (int captured by value, safe on GPU)
6. Update `amr/inputs/advection_*` to add the `adv.velocity_type` line to each

**Relevant files:**
- `extern/amrex/Tests/Amr/Advection_AmrCore/Source/face_velocity.H`
- `extern/amrex/Tests/Amr/Advection_AmrCore/Source/DefineVelocity.cpp`
- `extern/amrex/Tests/Amr/Advection_AmrCore/Source/AmrCoreAdv.H`
- `extern/amrex/Tests/Amr/Advection_AmrCore/Source/AmrCoreAdv.cpp` (ReadParameters)
- `amr/inputs/advection_periodic`, `advection_rotating`, `advection_vortex`
