import os
import glob
import subprocess
import re
import sys

def strip_ignores():
    files = glob.glob('src/**/*.py', recursive=True)
    for filepath in files:
        with open(filepath, 'r') as f:
            content = f.read()

        # Remove # type: ignore...
        # We use a regex that matches "  # type: ignore..." or "# type: ignore..."
        # And remove it.
        new_content = re.sub(r'(\s*)# type: ignore.*', '', content)

        # Also remove trailing whitespace resulting from removal
        # new_content = re.sub(r'[ \t]+$', '', new_content, flags=re.MULTILINE)

        if content != new_content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"Stripped ignores from {filepath}")

def parse_ty_output(output):
    errors = []
    lines = output.split('\n')
    current_code = None

    # Regex for code: error[code]: or warning[code]:
    # It might be colored, so we strip ANSI codes first?
    # Or regex allowing extra chars.

    for line in lines:
        # Strip ANSI codes
        line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

        m_code = re.match(r'^(error|warning)\[([a-zA-Z0-9-]+)\]:', line_clean)
        if m_code:
            current_code = m_code.group(2)
            continue

        m_loc = re.search(r'--> (src/[^:]+):(\d+):', line_clean)
        if m_loc and current_code:
            filepath = m_loc.group(1)
            lineno = int(m_loc.group(2))
            errors.append((filepath, lineno, current_code))
            # We don't reset current_code because sometimes multiple locations are printed?
            # But usually subsequent locations are "defined here".
            # We only care about the first one usually.
            # But if we capture all, we might ignore definition lines too?
            # "info: Function defined here --> file:line"
            # We should only capture if it's the error location.
            # The error location usually comes right after error message.
            # Ty output structure:
            # error[code]: message
            #    --> file:line
            # info: ...
            #    --> file:line

            # If the line starts with "info:", it's not the error location.
            # So we check if line starts with info.

        if line_clean.strip().startswith('info:'):
            current_code = None # Reset code to avoid associating info locations with error

    return errors

def apply_ignores(errors):
    file_errors = {}
    for fp, ln, code in errors:
        if fp not in file_errors:
            file_errors[fp] = {}
        if ln not in file_errors[fp]:
            file_errors[fp][ln] = set()
        file_errors[fp][ln].add(code)

    for filepath, errs in file_errors.items():
        if not os.path.exists(filepath):
            print(f"File {filepath} not found")
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for ln, codes in errs.items():
            idx = ln - 1
            if idx < len(lines):
                line = lines[idx].rstrip('\n')
                ignore_str = f"# type: ignore[{', '.join(sorted(codes))}]"
                lines[idx] = f"{line}  {ignore_str}\n"

        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"Applied ignores to {filepath}")

if __name__ == '__main__':
    print("Stripping ignores...")
    strip_ignores()

    print("Running ty...")
    # Ensure we use the env
    env = os.environ.copy()
    result = subprocess.run(['uv', 'run', 'ty', 'check', 'src/'], capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    print("Parsing output...")
    errors = parse_ty_output(output)
    print(f"Found {len(errors)} errors/warnings")

    print("Applying ignores...")
    apply_ignores(errors)
