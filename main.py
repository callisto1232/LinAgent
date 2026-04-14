import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

def print_ui_tree(obj, indent=0):
    # Only print things that have a name
    name = obj.get_name()
    role = obj.get_role_name()
    
    if name:
        print("  " * indent + f"[{role}] {name}")

    # Recursively look at children
    for i in range(obj.get_child_count()):
        child = obj.get_child_at_index(i)
        print_ui_tree(child, indent + 1)

def start_scan():
    Atspi.init()
    desktop = Atspi.get_desktop(0)
    
    print("--- Searching for Interactive Elements ---")
    for i in range(desktop.get_child_count()):
        app = desktop.get_child_at_index(i)
        
        # Focus on a specific app you have open, like Dolphin or Firefox
        if "dolphin" in app.get_name().lower() or "konsole" in app.get_name().lower():
            print(f"\nScanning App: {app.get_name()}")
            print_ui_tree(app)

if __name__ == "__main__":
    start_scan()
