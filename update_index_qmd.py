import os

file_path = '/Users/bartz/workspace/bart25t-desirability/bart25t/index.qmd'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Indices are 0-based, so line N is index N-1.

# Block 2 to remove: Lines 102-112 (inclusive)
# Index 101 to 111 (inclusive)
# 102: fit RandomForestRegressor models for each objective
# ...
# 112: ```
start_remove = 101
end_remove = 112 # slice upper bound is exclusive, so 112 means up to index 111

# Block 1 to replace: Lines 52-76 (inclusive)
# Index 51 to 75 (inclusive)
# 52: x1 = np.linspace...
# ...
# 76: plt.show()
start_replace = 51
end_replace = 76 # slice upper bound is exclusive, so 76 means up to index 75

# New content for Block 1
new_content = [
    "# fit RandomForestRegressor models for each objective\n",
    "models = []\n",
    "for i in range(y.shape[1]):\n",
    "    model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "    model.fit(X_base, y[:, i])\n",
    "    models.append(model)\n",
    "# calculate base Morris-Mitchell stats\n",
    "phi_base, J_base, d_base = mmphi_intensive(X_base, q=2, p=2)\n",
    "print(f\"phi_base: {phi_base}, J_base: {J_base}, d_base: {d_base}\")\n",
    "\n",
    "# Plot surfaces using mo_xy_surface\n",
    "from spotoptim.mo.pareto import mo_xy_surface\n",
    "mo_xy_surface(models, bounds=[(x_min, x_max), (x_min, x_max)], target_names=[\"Objective 1\", \"Objective 2\"])\n"
]

# Apply changes
# IMPORTANT: Delete the later block first to preserve indices of the earlier block
del lines[start_remove:end_remove]

# Now replace the earlier block
lines[start_replace:end_replace] = new_content

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Successfully updated index.qmd")
