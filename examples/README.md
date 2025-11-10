# SpotOptim Examples

This directory contains examples demonstrating how to use SpotOptim for various optimization problems.

## Available Examples

### 1. Aircraft Wing Weight Optimization (AWWE)

Two formats available:

#### Quarto Tutorial (`awwe.qmd`)

Comprehensive tutorial teaching surrogate-based optimization through the 9-dimensional Aircraft Wing Weight problem.

**Features:**
- Step-by-step explanation of each optimization stage
- Visualization of surrogate models
- Comparison of different surrogate types (GP vs Kriging)
- Sensitivity analysis
- Homework exercise with 10-dimensional problem

**To render:**
```bash
quarto render awwe.qmd
```

This will generate `awwe.html` with all code executed and plots generated.

**Prerequisites:**
- Quarto installed ([installation guide](https://quarto.org/docs/get-started/))
- spotoptim package installed

#### Python Script (`awwe_optimization.py`)

Standalone script demonstrating the optimization process.

**To run:**
```bash
python awwe_optimization.py
```

**Output:**
- Console output showing optimization progress
- `awwe_convergence.png` - Convergence plot
- `awwe_surrogate.png` - Surrogate model visualization

**Expected Results:**
- ~47% weight reduction from baseline (233.83 lb → ~124 lb)
- 70 function evaluations (20 initial + 50 optimization)
- Key findings: Reduce wing area, load factor, and design weight

### 2. Surrogate Plotting Demo (`plot_surrogate_demo.py`)

Simple demonstration of the `plot_surrogate()` method on a 2D sphere function.

**To run:**
```bash
python plot_surrogate_demo.py
```

Shows how to visualize:
- Prediction surface
- Uncertainty surface
- Contour plots with evaluated points

## Problem Descriptions

### Aircraft Wing Weight Example (AWWE)

**Objective:** Minimize the weight of an unpainted light aircraft wing

**Parameters (9D):**
1. Wing area (Sw)
2. Fuel weight (Wfw)
3. Aspect ratio (A)
4. Quarter-chord sweep (Lambda)
5. Dynamic pressure (q)
6. Taper ratio (lambda)
7. Thickness to chord ratio (Rtc)
8. Ultimate load factor (Nz)
9. Design gross weight (Wdg)

**Equation:**
```
W = 0.036 * Sw^0.758 * Wfw^0.0035 * (A/cos²(Lambda))^0.6 
    * q^0.006 * lambda^0.04 * (100*Rtc/cos(Lambda))^-0.3 
    * (Nz*Wdg)^0.49
```

**Baseline:** Cessna C172 Skyhawk (233.83 lb)

**Source:** Forrester et al. (2008), Engineering Design via Surrogate Modelling

## Learning Path

### Beginners
1. Start with `plot_surrogate_demo.py` to understand visualization
2. Read the Quarto tutorial `awwe.qmd` for theory and step-by-step guidance
3. Run `awwe_optimization.py` to see a complete workflow

### Advanced Users
1. Modify `awwe_optimization.py` to:
   - Add the 10th dimension (paint weight)
   - Try different acquisition functions
   - Compare multiple surrogate types
   - Add constraints (e.g., minimum wing area)

2. Create your own optimization problems:
   - Use `awwe_optimization.py` as a template
   - Replace the objective function
   - Adjust bounds and parameters
   - Add domain-specific visualizations

## Key Concepts Demonstrated

### Bayesian Optimization
- **Surrogate models**: Learn function behavior from samples
- **Acquisition functions**: Balance exploration vs exploitation
- **Initial design**: Space-filling samples using Latin Hypercube
- **Sequential optimization**: Iteratively select promising points

### SpotOptim Features
- **Multiple surrogates**: GP (default) and Kriging
- **Flexible configuration**: Initial samples, iterations, acquisition functions
- **Visualization**: `plot_surrogate()` method for 2D slices
- **Compatibility**: scipy-style results, scikit-learn interface

### Analysis Techniques
- **Convergence analysis**: Track best solution over iterations
- **Sensitivity analysis**: Identify important parameters
- **Surrogate validation**: Compare predictions to true function
- **Parameter comparison**: Understand changes from baseline

## Tips for Success

1. **Start Small**: Begin with 20-30 initial samples and 30-50 optimization iterations
2. **Visualize**: Use `plot_surrogate()` to understand the landscape
3. **Validate**: Check surrogate accuracy on test points
4. **Experiment**: Try different acquisition functions ('ei', 'y', 'pi')
5. **Document**: Record results and insights for comparison

## Homework Exercises

### Exercise 1: Add Paint Weight (Easy)
Extend the AWWE problem to 10 dimensions by adding paint weight parameter.

**Hints:**
- Paint weight: Wp ∈ [0.025, 0.08] lb/ft²
- Add term: + Sw * Wp to the equation
- Baseline: Wp = 0.064 lb/ft²

### Exercise 2: Constrained Optimization (Medium)
Add a constraint that wing area must be at least 160 ft² for structural reasons.

**Hints:**
- Modify objective to penalize violations
- Or filter out invalid solutions in `_suggest_next_point()`

### Exercise 3: Multi-Objective (Hard)
Minimize both weight AND cost, where cost = 50*Sw + 10*Wfw + ... (define your own)

**Hints:**
- Use weighted sum approach
- Or run two separate optimizations and compare Pareto front

## References

- Forrester, A., Sobester, A., & Keane, A. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley.
- SpotOptim GitHub: https://github.com/sequential-parameter-optimization/spotoptim
- SPOT methodology: Sequential Parameter Optimization Toolbox

## Contributing

Have an interesting optimization problem? Submit a PR with:
1. Python script demonstrating the problem
2. Brief description in this README
3. Expected results and insights
4. (Optional) Quarto tutorial for teaching

## Questions?

- Check the main README.md in the repository root
- Browse the test files in `tests/` for more usage examples
- Open an issue on GitHub
