#!/bin/zsh
# Convert markdown files from docs/manuals to Jupyter notebooks
# Usage: ./scripts/md_to_ipynb.sh

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    source "$VENV_ACTIVATE"
    echo "Virtual environment activated: $PROJECT_ROOT/.venv"
else
    echo "Warning: Virtual environment not found at $VENV_ACTIVATE"
    echo "Proceeding with system Python..."
fi

# Define paths
MANUALS_DIR="$PROJECT_ROOT/docs/manuals"
OUTPUT_DIR="$PROJECT_ROOT/docs/ipynb"

# Check if manuals directory exists
if [[ ! -d "$MANUALS_DIR" ]]; then
    echo "Error: Manuals directory not found: $MANUALS_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting markdown files to Jupyter notebooks..."
echo "Source directory: $MANUALS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to convert a single markdown file to ipynb
convert_md_to_ipynb() {
    local md_file="$1"
    local basename="$(basename "$md_file" .md)"
    local ipynb_file="$OUTPUT_DIR/${basename}.ipynb"
    
    echo "Converting: $basename.md -> $basename.ipynb"
    
    # Use jupytext to convert markdown to notebook
    # Install with: pip install jupytext
    if command -v jupytext &> /dev/null; then
        jupytext --to notebook --output "$ipynb_file" "$md_file"
        echo "  ✓ Created: $ipynb_file"
    else
        echo "  ✗ Error: jupytext not found. Install with: pip install jupytext"
        return 1
    fi
}

# Counter for statistics
total_files=0
converted_files=0
failed_files=0

# Process all markdown files in the manuals directory
for md_file in "$MANUALS_DIR"/*.md; do
    if [[ -f "$md_file" ]]; then
        total_files=$((total_files + 1))
        
        if convert_md_to_ipynb "$md_file"; then
            converted_files=$((converted_files + 1))
        else
            failed_files=$((failed_files + 1))
        fi
        echo ""
    fi
done

# Print summary
echo "======================================"
echo "Conversion Summary"
echo "======================================"
echo "Total files found:     $total_files"
echo "Successfully converted: $converted_files"
echo "Failed conversions:    $failed_files"
echo ""
echo "Output directory: $OUTPUT_DIR"

# Check if jupytext is installed, if not provide installation instructions
if ! command -v jupytext &> /dev/null; then
    echo ""
    echo "======================================"
    echo "INSTALLATION REQUIRED"
    echo "======================================"
    echo "jupytext is not installed."
    echo ""
    echo "To install jupytext, run:"
    echo "  source .venv/bin/activate"
    echo "  pip install jupytext"
    exit 1
fi

exit 0
