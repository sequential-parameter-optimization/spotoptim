
import ast
import sys
import os

# Define groups
GROUPS = {
    "Core": ["__init__", "__getattr__", "__setattr__", "__dir__", "__post_init__"],
    "Configuration & Helpers": ["_set_seed", "detect_var_type", "modify_bounds_based_on_var_type", "handle_default_var_trans", "process_factor_bounds", "get_best_hyperparameters", "_repair_non_numeric", "_reinitialize_components", "_get_pickle_safe_optimizer"],
    "Dimension Reduction": ["_setup_dimension_reduction", "to_red_dim", "to_all_dim"],
    "Variable Transformation": ["transform_value", "inverse_transform_value", "_transform_X", "_inverse_transform_X", "transform_bounds", "_map_to_factor_values"],
    "Initial Design": ["get_initial_design", "_generate_initial_design", "_curate_initial_design", "_rm_NA_values", "_validate_x0", "_check_size_initial_design", "_get_best_xy_initial_design", "_update_repeats_infill_points", "_remove_nan"],
    "Surrogate & Acquisition": ["_fit_surrogate", "_fit_scheduler", "_predict_with_uncertainty", "_acquisition_function", "_optimize_acquisition_tricands", "_optimize_acquisition_de", "_optimize_acquisition_scipy", "_try_optimizer_candidates", "_handle_acquisition_failure", "_try_fallback_strategy", "_get_shape", "_store_mo", "_mo2so", "_get_ranks", "_get_ocba", "_get_ocba_X", "_evaluate_function", "_select_distant_points", "_select_best_cluster", "_selection_dispatcher", "select_new", "acquisition", "optimize_acquisition_func"],
    "Optimization Loop": ["optimize", "_optimize_single_run", "suggest_next_infill_point", "_handle_NA_new_points", "_update_best_main_loop", "_determine_termination", "_apply_ocba", "_apply_penalty_NA"],
    "Storage & Statistics": ["_init_storage", "_update_storage", "update_stats", "_update_success_rate", "_get_success_rate", "_aggregate_mean_var"],
    "Results & Analysis": ["save_result", "load_result", "save_experiment", "load_experiment", "_get_result_filename", "_get_experiment_filename", "print_results", "print_best", "print_results_table", "print_design_table", "get_results_table", "get_design_table", "gen_design_table", "get_importance", "sensitivity_spearman", "get_stars"],
    "TensorBoard Integration": ["_clean_tensorboard_logs", "_init_tensorboard_writer", "_write_tensorboard_scalars", "_write_tensorboard_hparams", "_close_tensorboard_writer", "_init_tensorboard", "_close_and_del_tensorboard_writer"],
    "Plotting": ["plot_progress", "plot_surrogate", "plot_important_hyperparameter_contour", "_plot_surrogate_with_factors", "plot_importance", "plot_parameter_scatter"]
}

FILE_PATH = "src/spotoptim/SpotOptim.py"

def get_method_source(source_lines, node):
    # Determine start line (including decorators)
    start_lineno = node.lineno
    if node.decorator_list:
        start_lineno = min(d.lineno for d in node.decorator_list)
    
    # Check for preceding comments/docstrings that belong to the method?
    # A bit hard with AST, but usually they are inside the method body (docstrings).
    # External comments (hash) are ignored by AST.
    # We will assume standard formatting where methods are separated by blank lines.
    
    end_lineno = node.end_lineno
    
    # Extract lines (1-indexed to 0-indexed)
    method_lines = source_lines[start_lineno-1 : end_lineno]
    
    return "".join(method_lines)

def run():
    print(f"Reading {FILE_PATH}...")
    with open(FILE_PATH, "r") as f:
        source = f.read()
    
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    
    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SpotOptim":
            class_node = node
            break
            
    if not class_node:
        print("Error: SpotOptim class not found")
        sys.exit(1)
        
    print(f"Found SpotOptim class at line {class_node.lineno}")
    
    methods = {}
    method_nodes = {}
    
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = get_method_source(source_lines, node)
            method_nodes[node.name] = node
            
    # Check for uncategorized
    all_methods = set(methods.keys())
    categorized = set()
    for group_methods in GROUPS.values():
        categorized.update(group_methods)
        
    uncategorized = all_methods - categorized
    if uncategorized:
        print(f"Warning: The following methods are not categorized: {uncategorized}")
        # Add them to a new group "Others"
        GROUPS["Others"] = sorted(list(uncategorized))
        
    # Reconstruct class body
    # We keep everything BEFORE the first method as class header/docstring/attributes
    
    # Find the start of the first method to know where the header ends
    if not method_nodes:
        print("No methods found!")
        sys.exit(1)
        
    first_method_start = min(
        (n.lineno if not n.decorator_list else min(d.lineno for d in n.decorator_list)) 
        for n in method_nodes.values()
    )
    
    # The class header is everything from class_node.lineno to first_method_start - 1
    # BUT we need to be careful about matching indentation.
    
    # Actually, a safer way:
    # 1. Identify the Class Body Range in the original file.
    # 2. Extract Class Header (up to first method).
    # 3. Construct new body.
    # 4. Replace.
    
    # But class attributes might be interspersed.
    # We should extract NON-function nodes too?
    # No, usually attributes are at the top.
    # Let's check if there are any statements between methods.
    
    non_method_nodes = []
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            non_method_nodes.append(node)
            
    # If non_method_nodes exist after the first method, we might lose them.
    # Let's warn if we detect mixed content.
    last_attr_line = 0
    if non_method_nodes:
        last_attr_line = max(n.end_lineno for n in non_method_nodes)
        
    print(f"First method starts at {first_method_start}, last non-method at {last_attr_line}")
    if last_attr_line > first_method_start:
        print("Warning: There are class attributes or statements mixed with methods. They might be misplaced.")
        # We will try to preserve them by assuming they belong to the top if they are assignments.
        # But if they are between methods, this script is too simple.
        
    # Construct new body
    new_body_lines = []
    
    # Header: from class def start up to first method
    # Wait, source_lines includes the whole file. 
    # The file starts with imports, then class def.
    # We want to keep everything before the class starts? Yes.
    # And the class header.
    
    # Let's say we replace the content starting from first_method_start to the end of the class.
    # End of class: max(n.end_lineno for n in class_node.body)
    last_method_end = max(n.end_lineno for n in method_nodes.values())
    class_end = max(last_method_end, last_attr_line)
    
    # File preamble + Class Header
    # Everything up to first_method_start-1
    new_source = "".join(source_lines[:first_method_start-1])
    
    # Now append methods by group
    for group_name, method_list in GROUPS.items():
        # strict=False allows defined methods to be missing (e.g. if I made a typo in GROUPS)
        # But we should check.
        available_methods = [m for m in method_list if m in methods]
        
        if not available_methods:
            continue
            
        new_source += f"\n    # {'='*20}\n    # {group_name}\n    # {'='*20}\n"
        
        for method_name in available_methods:
            new_source += "\n" + methods[method_name] + "\n"
            
    # Append anything after the class? 
    # Usually the class ends the file or we have if __name__ == main.
    # We can append source_lines[class_end:]
    if class_end < len(source_lines):
        new_source += "".join(source_lines[class_end:])
        
    # Write to new file first for verification
    with open("src/spotoptim/SpotOptim_reordered.py", "w") as f:
        f.write(new_source)
        
    print("Written to src/spotoptim/SpotOptim_reordered.py")

if __name__ == "__main__":
    run()
