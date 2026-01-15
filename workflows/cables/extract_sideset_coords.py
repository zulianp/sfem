#!/usr/bin/env python3
"""
Automated script to extract coordinates from screenshots using OCR
and generate create_sideset commands.

Usage:
    python3 extract_sideset_coords.py [--sspics-dir SSPICS_DIR] [--radius RADIUS] [--output OUTPUT]
    python3 extract_sideset_coords.py --update-script cables.sh
"""

import re
import subprocess
import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def extract_coords_from_image(image_path: Path) -> Optional[Tuple[float, float, float]]:
    """Extract coordinates from an image using tesseract OCR."""
    try:
        result = subprocess.run(
            ['tesseract', str(image_path), 'stdout'],
            capture_output=True,
            text=True,
            check=True
        )
        text = result.stdout
        
        # Try to find coordinates in format: Coords: (x, y, z)
        pattern1 = r'Coords:\s*\(([-]?\d+\.?\d*),\s*([-]?\d+\.?\d*),\s*([-]?\d+\.?\d*)\)'
        match = re.search(pattern1, text)
        if match:
            return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        
        # Try to find coordinates in format: (x, y, z)
        pattern2 = r'\(([-]?\d+\.?\d*),\s*([-]?\d+\.?\d*),\s*([-]?\d+\.?\d*)\)'
        match = re.search(pattern2, text)
        if match:
            return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        
        # Try to find three consecutive floating point numbers
        pattern3 = r'([-]?\d+\.\d+)[,\s]+([-]?\d+\.\d+)[,\s]+([-]?\d+\.\d+)'
        matches = re.findall(pattern3, text)
        if matches:
            # Take the first match that looks like coordinates
            for match in matches:
                x, y, z = float(match[0]), float(match[1]), float(match[2])
                # Basic validation: coordinates should be in reasonable range
                if abs(x) < 10 and abs(y) < 10 and abs(z) < 10:
                    return (x, y, z)
        
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running tesseract on {image_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None


def scan_screenshots(sspics_dir: Path) -> List[Tuple[Path, Optional[Tuple[float, float, float]]]]:
    """Scan directory for PNG screenshots and extract coordinates."""
    results = []
    
    if not sspics_dir.exists():
        print(f"Error: Directory {sspics_dir} does not exist", file=sys.stderr)
        return results
    
    # Get all PNG files, sorted by name
    png_files = sorted(sspics_dir.glob('*.png'))
    
    if not png_files:
        print(f"No PNG files found in {sspics_dir}", file=sys.stderr)
        return results
    
    print(f"Found {len(png_files)} screenshot(s), extracting coordinates...", file=sys.stderr)
    
    for img_path in png_files:
        coords = extract_coords_from_image(img_path)
        results.append((img_path, coords))
        if coords:
            print(f"  {img_path.name}: {coords}", file=sys.stderr)
        else:
            print(f"  {img_path.name}: No coordinates found", file=sys.stderr)
    
    return results


def generate_create_sideset_commands(
    coords_list: List[Tuple[float, float, float]],
    mesh_name: str = "cables",
    radius: float = 0.997,
    start_index: int = 1,
    debug: bool = True
) -> List[str]:
    """Generate create_sideset commands from coordinates."""
    commands = []
    for idx, (x, y, z) in enumerate(coords_list, start=start_index):
        output_name = f"ex{idx}"
        debug_flag = "SFEM_DEBUG=1 " if debug else ""
        cmd = f"{debug_flag}create_sideset {mesh_name} {x} {y} {z} {radius} {output_name}"
        commands.append(cmd)
    return commands


def generate_raw_to_db_commands(
    num_sidesets: int,
    start_index: int = 1,
    coords_mesh: str = "cables",
    cell_type: str = "tri3"
) -> List[str]:
    """Generate raw_to_db.py commands for sidesets."""
    commands = []
    for idx in range(start_index, start_index + num_sidesets):
        output_name = f"ex{idx}"
        cmd = f"raw_to_db.py {output_name}/surf\t\t\t{output_name}/surf.vtk\t\t\t--coords={coords_mesh} --cell_type={cell_type}"
        commands.append(cmd)
    return commands


def update_script(
    script_path: Path,
    create_sideset_cmds: List[str],
    raw_to_db_cmds: List[str],
    radius: float
):
    """Update the cables.sh script with extracted coordinates."""
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist", file=sys.stderr)
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the section between set -x and raw_to_db.py commands
    # We'll replace the create_sideset commands section
    
    # Build the new create_sideset commands block
    create_sideset_block = "\t# create_sideset format: create_sideset <mesh> <x> <y> <z> <radius> <output_name>\n"
    create_sideset_block += "\t# Coordinates extracted from sspics screenshots (Use tesseract for OCR)\n"
    for cmd in create_sideset_cmds:
        create_sideset_block += f"\t{cmd}\n"
    
    # Find and replace the create_sideset section
    pattern = r'(set -x\s+# create_sideset format.*?)(?=\traw_to_db\.py|\t# Note:)'
    replacement = create_sideset_block.rstrip() + "\n"
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # If pattern not found, try to insert before raw_to_db commands
        pattern2 = r'(\traw_to_db\.py ex\d)'
        if re.search(pattern2, content):
            # Insert before first raw_to_db command
            content = re.sub(
                pattern2,
                create_sideset_block.rstrip() + "\n\n\t" + r'\1',
                content,
                count=1
            )
        else:
            print("Warning: Could not find insertion point in script", file=sys.stderr)
            return False
    
    # Update raw_to_db commands
    raw_to_db_block = ""
    for cmd in raw_to_db_cmds:
        raw_to_db_block += f"\t{cmd}\n"
    
    # Replace raw_to_db commands section
    pattern3 = r'(\traw_to_db\.py ex\d.*?)(?=\t\n|\t# |\tcd \$HERE)'
    if re.search(pattern3, content, re.DOTALL):
        content = re.sub(pattern3, raw_to_db_block.rstrip() + "\n", content, flags=re.DOTALL)
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {script_path}", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract coordinates from screenshots and generate create_sideset commands'
    )
    parser.add_argument(
        '--sspics-dir',
        type=str,
        default='sspics',
        help='Directory containing screenshot images (default: sspics)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=0.997,
        help='Radius for create_sideset (default: 0.997)'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='cables',
        help='Mesh name for create_sideset (default: cables)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=1,
        help='Starting index for sideset names (default: 1, so ex1, ex2, ...)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for commands (default: stdout)'
    )
    parser.add_argument(
        '--update-script',
        type=str,
        help='Update cables.sh script with extracted coordinates'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Do not include SFEM_DEBUG=1 in commands'
    )
    parser.add_argument(
        '--cell-type',
        type=str,
        default='tri3',
        help='Cell type for raw_to_db commands (default: tri3)'
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    sspics_dir = script_dir / args.sspics_dir
    
    # Scan screenshots
    results = scan_screenshots(sspics_dir)
    
    # Filter out None coordinates
    valid_coords = [(path, coords) for path, coords in results if coords is not None]
    
    if not valid_coords:
        print("Error: No valid coordinates found in any screenshot", file=sys.stderr)
        return 1
    
    # Extract just the coordinates
    coords_list = [coords for _, coords in valid_coords]
    
    # Generate commands
    create_sideset_cmds = generate_create_sideset_commands(
        coords_list,
        mesh_name=args.mesh,
        radius=args.radius,
        start_index=args.start_index,
        debug=not args.no_debug
    )
    
    raw_to_db_cmds = generate_raw_to_db_commands(
        len(coords_list),
        start_index=args.start_index,
        coords_mesh=args.mesh,
        cell_type=args.cell_type
    )
    
    # Output or update script
    if args.update_script:
        script_path = script_dir / args.update_script
        if update_script(script_path, create_sideset_cmds, raw_to_db_cmds, args.radius):
            print(f"\nSuccessfully updated {script_path}", file=sys.stderr)
            print(f"Generated {len(create_sideset_cmds)} create_sideset commands", file=sys.stderr)
            return 0
        else:
            return 1
    else:
        # Output to file or stdout
        output_lines = []
        output_lines.append("# create_sideset format: create_sideset <mesh> <x> <y> <z> <radius> <output_name>")
        output_lines.append("# Coordinates extracted from sspics screenshots (Use tesseract for OCR)")
        output_lines.extend(create_sideset_cmds)
        output_lines.append("")
        output_lines.extend(raw_to_db_cmds)
        
        output_text = "\n".join(output_lines) + "\n"
        
        if args.output:
            output_path = script_dir / args.output
            with open(output_path, 'w') as f:
                f.write(output_text)
            print(f"Commands written to {output_path}", file=sys.stderr)
        else:
            print(output_text)
        
        return 0


if __name__ == '__main__':
    sys.exit(main())

