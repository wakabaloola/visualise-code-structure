#!/usr/bin/env python3

import ast
import os
import sys
import fnmatch
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

error_messages = []


@contextmanager
def less_pager():
    """
    Context manager that captures output and displays it using 'less'.
    Falls back to regular printing if less is not available.
    """
    # Check if less is available
    if not subprocess.run(['which', 'less'], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE).returncode == 0:
        # If less is not available, just print normally
        yield
        return

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
        tmp_file_path = tmp_file.name
        # Store original stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # Redirect stdout and stderr to our temporary file
        sys.stdout = tmp_file
        sys.stderr = tmp_file
        
        try:
            yield
        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Close and reopen the file for reading
            tmp_file.flush()
            tmp_file.close()
            
            try:
                # Use less to display the content
                subprocess.run(['less', '-R', tmp_file_path], 
                             check=False,
                             stdout=old_stdout,
                             stderr=old_stderr)
            except KeyboardInterrupt:
                pass  # Handle Ctrl+C gracefully
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass


class FunctionClassVisitor(ast.NodeVisitor):
    def __init__(self, verbosity=0, show_docstrings=False):
        self.structure = {}
        self.current_class = None
        self.verbosity = verbosity
        self.show_docstrings = show_docstrings

    def get_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Extract detailed function information based on verbosity level,
        plus docstrings if requested.
        """
        
        # Basic argument gathering
        args_info = []
        for arg in node.args.args:
            # Always store both name and type internally,
            # so we can decide how to display them later.
            arg_type = None
            if hasattr(arg, 'annotation') and arg.annotation:
                arg_type = ast.unparse(arg.annotation)
            arg_info = {
                'name': arg.arg,
                'type': arg_type
            }
            args_info.append(arg_info)
            
        # Default values
        defaults = [ast.unparse(default) for default in node.args.defaults]
        if defaults:
            for i in range(-len(defaults), 0):
                args_info[i]['default'] = defaults[i]
                
        # Return type annotation
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns)
        
        # Docstring (only if show_docstrings is True)
        docstring = None
        if self.show_docstrings:
            docstring = ast.get_docstring(node)
        
        return {
            'name': node.name,
            'args': args_info,
            'returns': returns,
            'docstring': docstring
        }
    
    def format_function_signature(self, func_info: Dict[str, Any]) -> str:
        """Format function signature based on verbosity level."""
        v = self.verbosity

        # verbosity=0 => function name only
        if v == 0:
            return func_info['name']
        
        # For verbosity >= 1, we build some parentheses content
        if v in [1, 2]:
            args_str = []
            for arg in func_info['args']:
                # For -v1 => "argname[=default]"
                # For -v2 => "argname: type[=default]" (if type is available)
                parts = []
                
                # Always show argument name for v=1 or v=2
                parts.append(arg['name'])
                
                # For v=2, also show type if present
                if v == 2 and arg['type']:
                    parts.append(f": {arg['type']}")
                
                # Default?
                if 'default' in arg:
                    parts.append(f"={arg['default']}")
                
                args_str.append(''.join(parts))
            
            signature = f"{func_info['name']}({', '.join(args_str)})"
            
            # For v=2, also show return type if present
            if v == 2 and func_info['returns']:
                signature += f" -> {func_info['returns']}"
            
            return signature

        # verbosity=3 => only argument and return types
        if v == 3:
            # We keep the function name, but inside parentheses we only show the type or '?'.
            type_list = []
            for arg in func_info['args']:
                if arg['type']:
                    type_list.append(arg['type'])
                else:
                    type_list.append('?')
            
            signature = f"{func_info['name']}({', '.join(type_list)})"
            if func_info['returns']:
                signature += f" -> {func_info['returns']}"
            return signature

        # Fallback (shouldn't happen if we've covered all cases)
        return func_info['name']

    def visit_FunctionDef(self, node):
        func_info = self.get_function_info(node)
        formatted_signature = self.format_function_signature(func_info)
        
        # Create or update the data structure
        if self.current_class is None:
            if 'functions' not in self.structure:
                self.structure['functions'] = []
            self.structure['functions'].append({
                'signature': formatted_signature,
                'docstring': func_info['docstring']
            })
        else:
            if 'methods' not in self.structure[self.current_class]:
                self.structure[self.current_class]['methods'] = []
            self.structure[self.current_class]['methods'].append({
                'signature': formatted_signature,
                'docstring': func_info['docstring']
            })
        
    def visit_ClassDef(self, node):
        # Optionally show class docstring, too
        docstring = None
        if self.show_docstrings:
            docstring = ast.get_docstring(node)
        
        self.current_class = node.name
        self.structure[node.name] = {
            'methods': [],
            'docstring': docstring
        }
        self.generic_visit(node)
        self.current_class = None


class IgnorePatternManager:
    def __init__(self, ignore_patterns=None):
        self.ignore_patterns = ignore_patterns or []
        self._normalize_patterns()
    
    def _normalize_patterns(self):
        normalized = []
        for pattern in self.ignore_patterns:
            pattern = pattern.replace('\\', '/')
            if pattern.endswith('/'):
                pattern = pattern + '**/*'
            normalized.append(pattern)
        self.ignore_patterns = normalized
    
    def should_ignore(self, path):
        path_str = str(Path(path)).replace('\\', '/')
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            path_parts = Path(path_str).parts
            for i in range(len(path_parts)):
                partial_path = '/'.join(path_parts[:i+1])
                if fnmatch.fnmatch(partial_path, pattern):
                    return True
        return False

    def scan_these_dirs_only(self, path):
        # This method has not been implemented yet
        path_str = str(Path(path)).replace('\\', '/')
        for pattern in self.ignore_patters:
            ...

def get_code_structure(file_path: str, verbosity: int = 0, show_docstrings=False) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
            visitor = FunctionClassVisitor(verbosity=verbosity, show_docstrings=show_docstrings)
            visitor.visit(tree)
            return visitor.structure
        except Exception as e:
            error_messages.append(f"Error parsing {file_path}: {str(e)}")
            return {}


def print_structure(root_dir: str, verbosity: int = 0, ignore_patterns: Optional[List[str]] = None,
                    show_docstrings=False):
    """
    Print the code structure to stdout.
    
    Args:
        root_dir: Directory to analyze
        verbosity: Detail level 
                   (0=names only, 
                    1=with arg names, 
                    2=with arg names/types, 
                    3=only arg & return types)
        ignore_patterns: List of patterns to ignore
        show_docstrings: If True, prints docstrings for classes/functions.
    """
    global error_messages
    error_messages = []

    if ignore_patterns is None:
        ignore_patterns = [
            '**/__pycache__/**',
            '**/.git/**',
            '**/venv/**',
            '**/env/**',
            '**/.env/**',
            '**/.venv/**',
            '**/*.pyc',
            '**/.DS_Store',
        ]
    
    ignore_manager = IgnorePatternManager(ignore_patterns)
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not ignore_manager.should_ignore(os.path.join(root, d))]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            if ignore_manager.should_ignore(file_path):
                continue
            
            relative_path = os.path.relpath(file_path, root_dir)
            print(f"\n\033[1m{relative_path}\033[0m")  # Bold filename
            print("=" * len(relative_path))
            print()
            
            structure = get_code_structure(file_path, verbosity, show_docstrings)
            
            if 'functions' in structure and structure['functions']:
                print("\033[93mFunctions:\033[0m")  # Yellow heading
                print("-" * 10)
                for func in sorted(structure['functions'], key=lambda x: x['signature']):
                    print(f"└── {func['signature']}")
                    if show_docstrings and func['docstring']:
                        # Print docstring in faint text (gray) with indentation
                        doc_lines = func['docstring'].splitlines()
                        for line in doc_lines:
                            print(f"     \033[90m{line}\033[0m")
                print()
            
            classes = [k for k in structure.keys() if k != 'functions']
            if classes:
                print("\033[93mClasses:\033[0m")
                print("-" * 8)
                for class_name in sorted(classes):
                    class_info = structure[class_name]
                    print(f"└── {class_name}")
                    
                    # Class docstring?
                    if show_docstrings and class_info['docstring']:
                        doc_lines = class_info['docstring'].splitlines()
                        for line in doc_lines:
                            print(f"     \033[90m{line}\033[0m")
                    
                    # Methods
                    if 'methods' in class_info:
                        # Sort by signature
                        for method in sorted(class_info['methods'], key=lambda x: x['signature']):
                            print(f"    └── {method['signature']}")
                            if show_docstrings and method['docstring']:
                                doc_lines = method['docstring'].splitlines()
                                for line in doc_lines:
                                    print(f"         \033[90m{line}\033[0m")
                print()

    # After processing all files, if there were errors, print them
    if error_messages:
        print("\n\033[91mErrors encountered:\033[0m")  # Red color
        print("-" * 20)
        for error in error_messages:
            print(f"\033[90m{error}\033[0m")  # Gray color
        print()

def get_verbosity_level(args):
    """Determine verbosity level based on argument flags."""
    if args.types and args.arguments:
        return 2  # Both types and argument names
    elif args.types:
        return 3  # Only types, no argument names
    elif args.arguments:
        return 1  # Only argument names
    return 0     # Basic mode


def main():
    bold = "\033[1m"
    reset = "\033[0m"
    yellow = "\033[93m"

    # Create help text in man-page style
    help_text = f"""
{bold}CS(1)                                                    Code Structure Manual                                                    CS(1){reset}

{bold}NAME{reset}
       cs - Code structure analyzer for Python files

{bold}SYNOPSIS{reset}
       cs [-a] [-t] [-d] [--ignore PATTERNS...] [DIRECTORY]

{bold}DESCRIPTION{reset}
       Analyzes Python files in the specified directory (and its subdirectories) to display
       the structure of classes, methods, and functions. Can show argument names and type
       information with different combinations of flags.

{bold}OPTIONS{reset}
        DIRECTORY
            Directory to analyze. If not specified, uses the current directory.

        -a, --arguments
            Show argument names in function signatures

        -t, --types
            Show type hints and return types (or '?' when absent)

        -d, --docstrings
            Include docstrings in the output if present

        --ignore PATTERNS
            Specify patterns to ignore (e.g., "tests/**" "docs/**")
            Can provide multiple patterns

{bold}EXAMPLES{reset}
       {bold}cs .{reset}
              Recursively searches through all subdirectories (starting from the current directory '.') and returns the basic tree structure of python class names, method names and functions names present. (The '.' is optional.)

       {bold}cs -a /path/to/project{reset}
              Show class/method/function names, including 'argument' names. The search happens recursively through all (.py) files and subdirectories, starting at '/path/to/project'.

       {bold}cs -t /path/to/project{reset}
              Show class/method/function names, including argument and return 'types' (not including argument names). The search happens recursively through all (.py) files and subdirectories, starting at '/path/to/project'.

       {bold}cs -t -a /path/to/project{reset}
              Show class/method/function names, including argument types, return types, and argument names. The search happens recursively through all (.py) files and subdirectories, starting at '/path/to/project'. The format {bold}cs -ta /path/to/project{reset} is also acceptable.

       {bold}cs -d .{reset}
              Show class/method/function names, but in this case include any 'docstrings' that might be present. The search happens recursively through all (.py) files and subdirectories starting from the current directory, '.'.

       {bold}cs --ignore "tests/**" "docs/**" .{reset}
              Show class/method/function names, but 'ignore' any files in the the specified directories. The search happens recursively through all remaining (.py) files and subdirectories starting from the current directory, '.'.

{bold}EXIT STATUS{reset}
       0      Success
       1      Error occurred during execution

{bold}NOTES{reset}
       The analyzer ignores certain directories by default:
              - __pycache__
              - .git
              - venv, env, .env, .venv
              - Compiled Python files (.pyc)
    """

    # If help is requested, show the man-page style help
    if '-h' in sys.argv or '--help' in sys.argv:
        with less_pager():
            print(help_text)
        sys.exit(0)

    # Regular argument parsing for normal operation
    parser = argparse.ArgumentParser(
        prog='cs',
        usage=f"{bold}cs [OPTIONS] [DIRECTORY]{reset}",
        add_help=False  # Disable default help as we handle it above
    )

    parser.add_argument(
        'directory', 
        nargs='?', 
        default='.',
        help=f"{bold}Directory to analyze{reset} (default: current directory)"
    )

    parser.add_argument(
        '-a', 
        '--arguments', 
        action='store_true',
        help=f"{bold}Show argument names{reset}"
    )

    parser.add_argument(
        '-t', 
        '--types', 
        action='store_true',
        help=f"{bold}Show type hints and return types{reset}"
    )

    parser.add_argument(
        '--ignore', 
        nargs='*', 
        default=None,
        help=f"""{bold}Patterns to ignore{reset} (e.g. "tests/**" "docs/**")"""
    )

    parser.add_argument(
        '-d', 
        '--docstrings', 
        action='store_true',
        help=f"{bold}Include docstrings{reset} in the output (if present)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Use the pager context manager to enable scrollable output
    with less_pager():
        print_structure(
            root_dir=args.directory, 
            verbosity=get_verbosity_level(args), 
            ignore_patterns=args.ignore,
            show_docstrings=args.docstrings
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
