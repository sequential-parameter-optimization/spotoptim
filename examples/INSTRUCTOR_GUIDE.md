# Instructor Guide: Aircraft Wing Weight Optimization with SpotOptim

## Overview

This guide helps instructors teach surrogate-based optimization using the Aircraft Wing Weight Example (AWWE) and SpotOptim.

## Learning Materials

### For Students

1. **awwe.qmd** - Main tutorial (Quarto markdown)
   - Comprehensive step-by-step walkthrough
   - Executable code with explanations
   - Visualization of results
   - Duration: 2-3 hours for careful reading and execution

2. **awwe_optimization.py** - Standalone script
   - Quick demonstration of the optimization process
   - Generates plots automatically
   - Duration: 10-15 minutes to run and analyze

3. **Homework Exercise** - Included in awwe.qmd
   - Extends problem to 10 dimensions (adds paint weight)
   - Encourages independent problem-solving
   - Estimated time: 3-4 hours

### For Instructors

- **This guide** - Teaching tips and suggestions
- **examples/README.md** - Technical reference for all examples
- **Source code** - Fully documented and tested

## Course Integration

### Recommended Prerequisites

**Mathematics:**
- Calculus (derivatives, optimization basics)
- Linear algebra (matrices, vectors)
- Basic probability (Gaussian distributions)

**Programming:**
- Python basics (functions, arrays, loops)
- NumPy fundamentals
- Matplotlib for plotting

**Optimization:**
- Local vs global optimization concepts
- Gradient-based vs gradient-free methods
- Black-box function optimization

### Learning Objectives

After completing this module, students should be able to:

1. **Understand** surrogate-based optimization methodology
2. **Implement** optimization problems using SpotOptim
3. **Interpret** surrogate model visualizations
4. **Analyze** optimization convergence and results
5. **Apply** the methodology to new engineering problems
6. **Compare** different surrogate model types

## Teaching Approach

### Option 1: Lecture + Lab (Recommended)

**Week 1: Lecture (90 minutes)**
- Introduction to surrogate-based optimization (20 min)
- Problem formulation: AWWE (15 min)
- Bayesian optimization concepts (25 min)
- SpotOptim demonstration (20 min)
- Q&A (10 min)

**Week 2: Lab Session (120 minutes)**
- Students work through awwe.qmd tutorial (90 min)
- Discussion of results (15 min)
- Homework assignment explanation (15 min)

**Week 3: Homework Due**
- Students submit 10-dimensional problem solution
- Optional: Presentations of findings

### Option 2: Flipped Classroom

**Before Class:**
- Students read awwe.qmd (sections 1-6)
- Students run awwe_optimization.py
- Students prepare questions

**In Class (90 minutes):**
- Q&A on concepts (20 min)
- Advanced topics discussion (30 min)
- Group work on variations (30 min)
- Summary and homework (10 min)

### Option 3: Self-Paced Online

**Module Structure:**
1. Video introduction (10 min, instructor-created)
2. awwe.qmd tutorial (self-paced)
3. Quiz on key concepts (10 questions)
4. Homework assignment
5. Peer review of results

## Key Concepts to Emphasize

### 1. Surrogate Models

**What to teach:**
- Surrogate ≈ "cheap approximation" of expensive function
- Built from evaluated points
- Provides both prediction and uncertainty

**Common misconceptions:**
- "Surrogate is always accurate" → No, it's approximate
- "More samples = always better" → Diminishing returns exist
- "Surrogate finds global optimum" → No guarantees, but good heuristic

**Teaching tip:**
Use the surrogate visualization (Step 7) to show:
- High uncertainty in unexplored regions (white/yellow in uncertainty plot)
- Low uncertainty near evaluated points (dark blue)
- How surrogate surface approximates true function

### 2. Exploration vs Exploitation

**What to teach:**
- Exploitation: Search near known good points
- Exploration: Try uncertain regions
- Acquisition function balances both

**Activity:**
Show the convergence plot and ask students:
- When does exploration occur? (Early iterations, diverse samples)
- When does exploitation dominate? (Late iterations, focused search)
- How does acquisition='ei' achieve this balance?

### 3. Dimensionality Challenges

**What to teach:**
- 9D space is HUGE (billions of combinations)
- Visualization requires slicing (fix 7 dims, plot 2)
- Curse of dimensionality affects sampling

**Demonstration:**
Calculate search space volume:
```python
import numpy as np
bounds = [(150,200), (220,300), (6,10), (-10,10), (16,45), 
          (0.5,1), (0.08,0.18), (2.5,6), (1700,2500)]
volume = np.prod([b[1]-b[0] for b in bounds])
print(f"Search space volume: {volume:.2e}")
# Result: ~1.4e12 combinations!
```

With 70 evaluations, we sample ~5e-11 of the space!

### 4. Parameter Sensitivity

**What to teach:**
- Not all parameters equally important
- Sensitivity analysis identifies key drivers
- Design decisions should focus on important parameters

**Activity:**
Have students examine the sensitivity bar chart (Step 9) and discuss:
- Which parameters have largest impact?
- Why might load factor (Nz) be important?
- What engineering insights can we gain?

## Discussion Questions

### Beginner Level

1. What is a surrogate model and why do we use it?
2. How many function evaluations did we use? Why not use 1000?
3. What does the red line in the convergence plot represent?
4. Why did wing weight decrease by 47%?

### Intermediate Level

5. Compare initial exploration phase vs later exploitation phase
6. Explain the difference between GP and Kriging surrogates
7. How would you know if the surrogate is accurate enough?
8. What happens if we use only 5 initial samples instead of 20?

### Advanced Level

9. Design a modified acquisition function for your own balance of exploration/exploitation
10. How would you handle constraints (e.g., minimum wing area for safety)?
11. Can this method guarantee finding the global optimum? Why or why not?
12. How does this compare to genetic algorithms or gradient descent?

## Common Student Questions

### "Why not just try all combinations?"

**Answer:**
With 9 dimensions and reasonable discretization (say 10 values per dimension), 
that's 10^9 = 1 billion evaluations. At 1 second per evaluation, that's ~32 years!
Surrogate-based optimization finds good solutions with ~70 evaluations (~1 minute).

### "How do I know if my solution is good?"

**Answer:**
1. Compare to baseline (we got 47% improvement)
2. Check convergence (are we still improving?)
3. Validate with domain experts (does it make physical sense?)
4. Try multiple random seeds (is solution consistent?)

### "What if the surrogate is wrong?"

**Answer:**
That's okay! We evaluate the TRUE function at each point, not the surrogate.
The surrogate just helps us decide WHERE to evaluate next. Even if prediction
is off, we discover that and update the surrogate.

### "Can I use this for my problem?"

**Answer:**
Yes, if your problem has:
1. Expensive function evaluations (>1 second each)
2. Continuous or mixed variables
3. No gradients available
4. Limited evaluation budget (<1000 calls)

## Homework Grading Rubric

### 10-Dimensional Problem (100 points)

**Implementation (40 points)**
- Correct function modification (+20)
- Proper bounds and variable names (+10)
- Code runs without errors (+10)

**Analysis (40 points)**
- Optimal configuration reported (+10)
- Comparison with 9D results (+10)
- Sensitivity analysis included (+10)
- Quality of visualizations (+10)

**Insights (20 points)**
- Engineering interpretation (+10)
- Discussion of parameter interactions (+5)
- Suggestions for improvement (+5)

### Bonus Opportunities (+10 points each)

- Try different acquisition functions and compare
- Implement constraints
- Compare 3+ different surrogate models
- Create interactive visualization
- Write up as mini research paper

## Extended Activities

### Project Ideas

1. **Multi-Objective Optimization**
   - Minimize weight AND cost simultaneously
   - Plot Pareto front
   - Discuss trade-offs

2. **Robust Design**
   - Add noise/uncertainty to parameters
   - Find solution that's robust to variations
   - Compare to nominal optimization

3. **Constrained Optimization**
   - Add structural constraints (e.g., min wing area)
   - Implement penalty method
   - Compare constrained vs unconstrained

4. **Algorithm Comparison**
   - Compare SpotOptim vs scipy.optimize.differential_evolution
   - Compare SpotOptim vs random search
   - Analyze convergence rates

### Guest Speaker Suggestions

- Aerospace engineer (wing design)
- ML engineer (Bayesian optimization)
- Software developer (SpotOptim contributor)

## Assessment Ideas

### Quiz Questions (Multiple Choice)

1. What is the primary advantage of surrogate-based optimization?
   - a) Guaranteed global optimum
   - b) Faster than gradient descent
   - c) **Efficient for expensive evaluations**
   - d) Works only for convex functions

2. In Bayesian optimization, what does "acquisition function" do?
   - a) Purchases computing resources
   - b) **Selects next point to evaluate**
   - c) Calculates objective value
   - d) Cleans input data

3. Which parameter had the largest impact on wing weight?
   - (Varies by run, but typically Nz, Sw, or Wdg)

### Short Answer

1. Explain in 2-3 sentences how a surrogate model helps optimization.
2. Why did we use 20 initial samples? What would happen with 5 or 100?
3. Describe one way to validate that your optimized design is realistic.

### Coding Challenge

Modify the code to:
1. Use acquisition='pi' instead of 'ei'
2. Plot the difference in convergence
3. Explain which is better and why

## Additional Resources

### For Students
- SpotOptim documentation
- Forrester et al. (2008) textbook chapters 1-3
- [Distill.pub article on Bayesian optimization](https://distill.pub/2020/bayesian-optimization/)

### For Instructors
- Original AWWE paper (Forrester et al., 2008)
- Sequential Parameter Optimization Toolbox (SPOT) methodology
- spotpython package (more advanced features)

## Technical Support

### Common Issues

**"Import error: No module named spotoptim"**
- Solution: `pip install spotoptim` or `uv pip install spotoptim`

**"Quarto render fails"**
- Solution: Install Quarto from https://quarto.org/
- Check Python environment is accessible to Quarto

**"Plots don't show up"**
- Solution: Check matplotlib backend (use `%matplotlib inline` in Jupyter)
- Or save plots to file with `plt.savefig()`

**"Optimization gets stuck"**
- Solution: Try different random seed
- Increase `max_iter` or `n_initial`
- Check for function evaluation errors

### Getting Help

- GitHub Issues: https://github.com/sequential-parameter-optimization/spotoptim/issues
- Email maintainer (check repository)
- SpotOptim discussion forum (if available)

## Feedback Welcome!

This teaching material is continuously improved. Please share:
- What worked well in your class
- What students found confusing
- Suggestions for additional examples
- Errors or typos

Contact: [Repository maintainer]

---

**Version:** 1.0  
**Last Updated:** November 2025  
**Author:** SpotOptim Development Team
