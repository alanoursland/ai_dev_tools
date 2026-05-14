import os
import ast

def get_definitions(root_dir):
    summary = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, root_dir)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())

                    file_defs = []
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            # 1. Capture Inheritance
                            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                            base_str = f"({', '.join(bases)})" if bases else ""
                            
                            methods = []
                            for n in node.body:
                                if isinstance(n, ast.FunctionDef):
                                    # Allow __init__ and important dunder methods, filter single-underscore privates
                                    important_dunders = ['__init__', '__getitem__', '__setitem__', '__contains__',
                                                        '__eq__', '__lt__', '__le__', '__gt__', '__ge__',
                                                        '__add__', '__sub__', '__mul__', '__repr__', '__str__']
                                    if not n.name.startswith("_") or n.name in important_dunders:
                                        # Optional: Capture args (simple version)
                                        args = [a.arg for a in n.args.args if a.arg != 'self']
                                        method_sig = f"{n.name}({', '.join(args)})"
                                        methods.append(method_sig)

                            file_defs.append(f"  class {node.name}{base_str}: {', '.join(methods)}")

                        elif isinstance(node, ast.FunctionDef):
                            if not node.name.startswith("_"):
                                # Capture top-level function args
                                args = [a.arg for a in node.args.args]
                                file_defs.append(f"  def {node.name}({', '.join(args)})")
                    if file_defs:
                        summary.append(f"\nFILE: {rel_path}")
                        summary.extend(file_defs)
                except:
                    continue
    return "\n".join(summary)

if __name__ == "__main__":
    print("Scanning source code")
    output = get_definitions("../src/kinopulse") # Point this to your src folder
    with open("library_structure.txt", "w") as f:
        f.write(output)
    print("Done. Upload 'library_structure.txt'.")
