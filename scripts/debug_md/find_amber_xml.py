import openmm.app as app
import os

try:
    # Try to find data directory
    app_dir = os.path.dirname(app.__file__)
    data_dir = os.path.join(app_dir, 'data')
    print(f"App Dir: {app_dir}")
    print(f"Data Dir: {data_dir}")
    
    # Recursive search for amber14-all.xml
    found = False
    for root, dirs, files in os.walk(data_dir):
        if 'amber14-all.xml' in files:
            fpath = os.path.join(root, 'amber14-all.xml')
            print(f"Found: {fpath}")
            with open(fpath, 'r') as f:
                print(f.read())
            found = True
            break
            
    if not found:
        print("amber14-all.xml not found in data dir.")
        
except Exception as e:
    print(f"Error: {e}")
