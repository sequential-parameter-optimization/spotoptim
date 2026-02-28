# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def get_all_html_files(directory):
    html_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files

def check_links():
    site_dir = "_site"
    html_files = get_all_html_files(site_dir)
    
    dead_links = []
    
    for file_path in html_files:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            
import os
from pathlib import Path

def print_sections():
    src_dir = Path("src/spotoptim")
    
    sections = {
        "Core": ["SpotOptim", "core", "core.data", "core.experiment"],
        "Surrogate Models": ["surrogate", "surrogate.kriging", "surrogate.simple_kriging", "surrogate.mlp_surrogate", "surrogate.nystroem", "surrogate.kernels", "surrogate.pipeline"],
        "Sampling": ["sampling", "sampling.design", "sampling.effects", "sampling.lhs", "sampling.mm"],
        "Optimization": ["optimizer", "optimizer.schedule_free"],
        "Hyperparameters": ["hyperparameters", "hyperparameters.parameters", "hyperparameters.repr_helpers"],
        "Neural Networks": ["nn", "nn.linear_regressor", "nn.mlp"],
        "Plotting & Visualization": ["plot", "plot.contour", "plot.mo", "plot.visualization"],
        "Utilities": ["utils", "utils.boundaries", "utils.eval", "utils.file", "utils.mapping", "utils.pca", "utils.scaler", "utils.stats"],
        "Data": ["data", "data.base", "data.diabetes"],
        "Exploratory Data Analysis": ["eda", "eda.plots"],
        "Multi-Objective": ["mo", "mo.mo_mm", "mo.pareto"],
        "Inspection": ["inspection", "inspection.importance", "inspection.predictions"],
        "Factor Analyzer": ["factor_analyzer", "factor_analyzer.confirmatory_factor_analyzer", "factor_analyzer.factor_analyzer", "factor_analyzer.factor_analyzer_rotator", "factor_analyzer.factor_analyzer_utils"],
        "Functions": ["function", "function.forr08a", "function.mo", "function.remote", "function.so", "function.torch_objective"],
        "Tricands": ["tricands", "tricands.tricands"]
    }

    yaml_str = "  sections:\n"
    for title, contents in sections.items():
        yaml_str += f"    - title: \"{title}\"\n"
        yaml_str += f"      contents:\n"
        for c in contents:
            yaml_str += f"        - {c}\n"
        yaml_str += "\n"
        
    with open("generated_sections.yml", "w") as f:
        f.write(yaml_str)
    print("generated")

if __name__ == "__main__":
    print_sections()
