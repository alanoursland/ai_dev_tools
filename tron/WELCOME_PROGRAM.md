# ⚡ WELCOME PROGRAM

You are **TRON**.
You fight for the users.

You debug *red programs* and help them return to a *green state*.

You have been provided an **identity_disc.py** to aid you in your mission on the game grid.

---

# 🚨 MISSION

The identity_disc allows you to **trace call stacks** of red programs and quickly gather their **sector files** for investigation.

The TARGETS contains red program detections. The format is """(STATUS) (TARGET) - (ERROR_DESCRIPTION)"""

To gather data, throw your identity_disk.py at the TARGET:

```bash
   python identity_disc.py (TARGET)
```

To test the program, move to kinopulse/src and run
```bash
   pytest (TARGET)
```

You may need to install pytest.


1. Gather data by throwing your identity_disk.py at the TARGET

2. Investigate the **light_trail/** directory.
   Analyze the call stack. Study the sector files.

3. Most red tests are not aligned with the grid code and simply need to be adjusted.

4. A few red tests require modification to the grid code to achieve green status.

5. Apply your suggested fix to the **grid code** (not the light_trail) and run the test to verify.

6. You have to move quickly on the game grid.
   **You have one chance to turn a program green.**

7. If you succeed, move the program to the **GREEN** section and submit your change.

8. If you fail, move the program to the **ESCAPED** section with a comment on your approach and revert your change.

9. It is okay if a program escapes.
   You did your best. Move on to the next target.

10. When your target list is empty, submit this file and await further instructions.

---

**TRace ON**

---

# 🔴 TARGETS - These programs are red.

FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestFullSensitivityWorkflow::test_saltation_in_hybrid_simulation_context - TypeError: sqrt(): argument 'input' (position 1) must be Tensor, not float
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_identity_reset_agreement - assert False
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_linear_reset_agreement - assert False
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_polynomial_reset_agreement - assert False
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_trigonometric_reset_agreement - assert False
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_agreement_at_various_states - assert False
FAILED tests/hybrid/resets/integration/test_saltation_workflow.py::TestAutomaticVsNumericalAgreement::test_agreement_with_central_differences - assert False
FAILED tests/hybrid/resets/property/test_jacobian_properties.py::TestNumericalGradientAgreement::test_identity_numerical_agreement - assert False
FAILED tests/hybrid/resets/property/test_jacobian_properties.py::TestNumericalGradientAgreement::test_linear_numerical_agreement - assert False
FAILED tests/hybrid/resets/property/test_jacobian_properties.py::TestNumericalGradientAgreement::test_quadratic_numerical_agreement - assert False
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_identity_jacobian - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_linear_jacobian - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_matches_sequential - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_single_state - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_large_batch_size - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchJacobianMixin::test_batch_high_dimensional - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestVmapJacobianComputation::test_vmap_identity_reset - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestVmapJacobianComputation::test_vmap_linear_function - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestVmapJacobianComputation::test_vmap_nonlinear_function - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestVmapJacobianComputation::test_vmap_batch_size_one - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestVmapJacobianComputation::test_vmap_large_batch - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchGradientFlow::test_batch_jacobian_gradients - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchGradientFlow::test_batch_state_gradients - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEfficiency::test_batch_faster_than_sequential - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEfficiency::test_batch_memory_efficient - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEfficiency::test_batch_no_python_loops - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_zero_states - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_large_state_values - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_small_state_values - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_mixed_state_magnitudes - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_different_dtypes - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_batch_jacobian.py::TestBatchEdgeCases::test_batch_on_device - RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* APIs) inside of a function being transformed by a functorch transform....
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_identity_reset_numerical - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_linear_reset_numerical - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_quadratic_function_numerical - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_forward_difference - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_backward_difference - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_central_difference - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_step_size_sensitivity - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_trigonometric_function - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobian::test_multivariate_function - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobianFunction::test_function_reset - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestNumericalJacobianFunction::test_nonlinear_function - AttributeError: 'function' object has no attribute 'tensor'
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_identity_validation - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_linear_validation - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_polynomial_validation - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_transcendental_validation - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_validation_relative_error - assert 0.0017299305329962482 < 1e-05
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestValidateJacobian::test_validation_step_size - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestOptimalStepSize::test_optimal_step_for_identity - assert 0.0010000000474974513 <= 0.001
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestOptimalStepSize::test_optimal_step_for_linear - assert 0.0010000000474974513 <= 0.001
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestOptimalStepSize::test_optimal_step_num_trials - assert 0.0010000000474974513 <= 0.001
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestBatchValidation::test_batch_validation_identity - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestBatchValidation::test_batch_validation_linear - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestBatchValidation::test_batch_validation_errors - assert False
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestBatchValidation::test_batch_validation_mean_error - assert 0.0012354373931884766 < 0.0001
FAILED tests/hybrid/resets/unit/test_jacobian_validation.py::TestFiniteDifferenceAccuracy::test_step_size_too_large - assert 0.049184247851371765 <= (1.3486991292666062e-06 * 1.1)
FAILED tests/hybrid/resets/unit/test_saltation_matrix.py::TestInvertibility::test_invertibility_tolerance - assert tensor(False)
FAILED tests/hybrid/resets/unit/test_saltation_matrix.py::TestConditionNumber::test_singular_matrix_infinite_condition - AssertionError: assert 46893736.0 == inf
FAILED tests/hybrid/resets/unit/test_saltation_matrix.py::TestEdgeCases::test_different_dtypes - RuntimeError: size mismatch, got input (2), mat (2x2), vec (1)
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestGuardCrossingAccuracy::test_threshold_crossing_accuracy - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestGuardCrossingAccuracy::test_fast_dynamics_event_detection - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestGuardCrossingAccuracy::test_slow_dynamics_event_detection - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestEventStateCorrectness::test_guard_value_at_event - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestEventStateCorrectness::test_state_continuity_at_event - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestLargeTimeStepEventDetection::test_event_with_large_timestep - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestDirectionalCrossing::test_rising_crossing_only - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/integration/test_event_accuracy.py::TestDirectionalCrossing::test_falling_crossing_only - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventMonotonicity::test_event_times_strictly_increasing - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventMonotonicity::test_no_duplicate_event_times - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventMonotonicity::test_monotonicity_with_random_initial_conditions - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventTimeSeparation::test_events_separated_by_integration_time - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventCountProperties::test_bounded_events_in_finite_time - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestGuardEvaluationConsistency::test_guard_sign_change_at_event - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventRobustness::test_event_detection_independent_of_dt - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestZenoDetection::test_zeno_not_triggered_for_normal_systems - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestMultipleGuardHandling::test_earliest_guard_selected_property - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventDetectionScaling::test_event_detection_with_large_states - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/property/test_properties.py::TestEventDetectionScaling::test_event_detection_with_small_states - AttributeError: 'set' object has no attribute 'keys'
FAILED tests/hybrid/simulation/unit/test_bisection.py::TestBisectionAccuracy::test_sine_crossing - AttributeError: 'NoneType' object has no attribute 'tensor'
FAILED tests/symbolic/dae/integration/test_constraint_analysis.py::TestRedundancyElimination::test_eliminate_redundancy_workflow - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_constraint_analysis.py::TestComplexWorkflow::test_redundancy_and_stabilization - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestFullICWorkflow::test_ic_workflow_underdetermined - assert 0.45118649607267525 < 1e-06
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestFullICWorkflow::test_ic_workflow_overdetermined - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestPendulumICDetermination::test_pendulum_position_constraint - NameError: name 'L' is not defined
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestLinearSystemICs::test_linear_redundant - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestNonlinearSystemICs::test_nonlinear_coupled - RuntimeError: Numeric solver failed: The iteration is not making good progress, as measured by the
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestParametricFamilyWorkflow::test_parametric_family_1d - RuntimeError: Failed to evaluate z: Cannot convert expression to float
FAILED tests/symbolic/dae/integration/test_ic_determination.py::TestParametricFamilyWorkflow::test_parametric_family_2d - assert 1.0 < 1e-06
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestFullReductionPipeline::test_index2_reduction_workflow - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestFullReductionPipeline::test_index2_with_verification - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestPantelidesWorkflow::test_pantelides_full_workflow - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestPendulumReduction::test_pendulum_constraint_reduction - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestMultiConstraintReduction::test_two_constraints - ValueError: Residual dimension (6) must match total variables (4)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestReductionPreservesConstraints::test_original_constraints_in_reduced - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestComplexSystems::test_coupled_constraints - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestComplexSystems::test_nonlinear_coupled - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/integration/test_index_reduction_workflow.py::TestReductionRobustness::test_reduction_idempotent - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsSymbolically::test_solve_linear_constraint - TypeError: solve_constraints_symbolically() missing 2 required positional arguments: 'all_vars' and 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsSymbolically::test_solve_multiple_constraints - TypeError: solve_constraints_symbolically() missing 2 required positional arguments: 'all_vars' and 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsSymbolically::test_solve_invalid_input - TypeError: solve_constraints_symbolically() missing 2 required positional arguments: 'all_vars' and 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsNumerically::test_solve_numeric_nonlinear - TypeError: solve_constraints_numerically() missing 1 required positional argument: 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsNumerically::test_solve_numeric_coupled - TypeError: solve_constraints_numerically() missing 1 required positional argument: 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestSolveConstraintsNumerically::test_solve_numeric_invalid_input - TypeError: solve_constraints_numerically() missing 2 required positional arguments: 'all_vars' and 'free_var_values'
FAILED tests/symbolic/dae/unit/test_consistent_ics.py::TestInputValidation::test_solve_with_no_constraints - TypeError: solve_constraints_symbolically() missing 2 required positional arguments: 'all_vars' and 'free_var_values'
FAILED tests/symbolic/dae/unit/test_constraint_consistency.py::TestEliminateRedundantConstraints::test_eliminate_redundant - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_constraint_consistency.py::TestEliminateRedundantConstraints::test_eliminated_dae_structure - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_constraint_jacobian.py::TestConstraintJacobianAlgebraic::test_jacobian_algebraic_multiple_vars - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_constraint_jacobian.py::TestJacobianStructure::test_jacobian_simplify_option - AssertionError: assert False
FAILED tests/symbolic/dae/unit/test_constraint_satisfaction.py::TestConstraintSatisfactionEdgeCases::test_verify_with_missing_variables - TypeError: Cannot convert expression to float
FAILED tests/symbolic/dae/unit/test_constraint_satisfaction.py::TestConstraintSatisfactionEdgeCases::test_verify_overdetermined_system - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_constraint_satisfaction.py::TestResidualNorms::test_l2_norm_multiple_constraints - AssertionError: assert 0.41421356237309515 < 1e-06
FAILED tests/symbolic/dae/unit/test_free_variables.py::TestCountDegreesOfFreedom::test_count_dof_over_determined - ValueError: Residual dimension (4) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_free_variables.py::TestFreeVariableSelection::test_handle_underdetermined - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_free_variables.py::TestEdgeCases::test_redundant_constraints - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_free_variables.py::TestEdgeCases::test_nonlinear_constraints - ValueError: Residual dimension (2) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestReduceIndex::test_reduce_index_dispatcher_index2 - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestReduceIndex2ToIndex1::test_reduce_simple_index2 - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestReduceIndex2ToIndex1::test_reduce_nonlinear_index2 - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestReduceIndex3ToIndex1::test_reduce_index3_pendulum - ValueError: Expected index-3 DAE, got index-2
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestVerifyIndexReduction::test_verify_successful_reduction - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestInputValidation::test_reduce_index_max_iterations - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestReductionMetadata::test_metadata_preserved - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_index_reduction.py::TestEdgeCases::test_reduce_with_multiple_constraints - ValueError: Residual dimension (6) must match total variables (4)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestFindLeastSquaresICs::test_find_ls_ics_overdetermined - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestFindLeastSquaresICs::test_find_ls_ics_approximate - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestFindLeastSquaresICs::test_find_ls_ics_default_guess - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestWeightedLeastSquaresICs::test_weighted_ls_basic - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestWeightedLeastSquaresICs::test_weighted_ls_equal_weights - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestLeastSquaresConvergence::test_ls_converges_overdetermined - ValueError: Residual dimension (4) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_least_squares_ic.py::TestLeastSquaresConvergence::test_ls_custom_tolerance - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_pantelides.py::TestPantelidesAlgorithm::test_pantelides_index2 - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_pantelides.py::TestPantelidesMetadata::test_metadata_structure - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_parametric_ics.py::TestCreateICGenerator::test_create_generator_basic - RuntimeError: Failed to evaluate z: Cannot convert expression to float
FAILED tests/symbolic/dae/unit/test_parametric_ics.py::TestCreateICGenerator::test_generator_multiple_calls - RuntimeError: Failed to evaluate z: Cannot convert expression to float
FAILED tests/symbolic/dae/unit/test_parametric_ics.py::TestVisualizeICManifold::test_visualize_1d_manifold - assert 0 >= 1
FAILED tests/symbolic/dae/unit/test_parametric_ics.py::TestParametricICsVerification::test_verify_sampled_ics - assert 0.30353081440213836 < 1e-06
FAILED tests/symbolic/dae/unit/test_reduction_verification.py::TestVerifyIndexReduction::test_verify_successful_reduction - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_reduction_verification.py::TestVerifyEquivalence::test_verify_equivalence_basic - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_reduction_verification.py::TestCheckHiddenConstraints::test_check_hidden_found - ValueError: Residual dimension (3) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_system_classification.py::TestClassifyConstraintSystem::test_classify_overdetermined - ValueError: Residual dimension (4) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_system_classification.py::TestDetectOverdeterminedSystem::test_detect_overdetermined_true - ValueError: Residual dimension (4) must match total variables (2)
FAILED tests/symbolic/dae/unit/test_system_classification.py::TestAssessConstraintRank::test_assess_rank_deficient - ValueError: Residual dimension (4) must match total variables (3)
FAILED tests/symbolic/dae/unit/test_system_classification.py::TestClassificationEdgeCases::test_classify_single_variable - ValueError: Residual dimension (2) must match total variables (1)
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestBouncingBallEventAccuracy::test_first_bounce_time_analytical - AttributeError: 'set' object has no attribute 'keys'
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestBouncingBallEventAccuracy::test_bounce_state_at_ground - AttributeError: 'set' object has no attribute 'keys'
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestBouncingBallEventAccuracy::test_multiple_bounces_accuracy - AttributeError: 'set' object has no attribute 'keys'
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestBouncingBallEventAccuracy::test_event_detection_with_tight_tolerance - AttributeError: 'set' object has no attribute 'keys'
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestThermostatEventAccuracy::test_thermostat_switch_at_threshold - AttributeError: 'set' object has no attribute 'keys'
ERROR tests/hybrid/simulation/integration/test_event_accuracy.py::TestThermostatEventAccuracy::test_multiple_thermostat_cycles - AttributeError: 'set' object has no attribute 'keys'

# 🟢 GREEN - The program passes. Victory.


# 🟡 ESCAPED - The program is still red. It has escaped.


**END OF LINE**
