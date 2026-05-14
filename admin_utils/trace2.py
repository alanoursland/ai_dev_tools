import sys
import ast
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, TextIO

# Reuse infrastructure
import call_tree
from python_semantic_graph import (
    PythonSemanticGraph,
    GraphBuilder,
    FileNode,
    HeuristicResolver,
    Call,
    Assignment,
    ControlFlow,
    LogicBlock,
    Keyword,
    TryBlock,
    CatchBlock
)

# =============================================================================
# UTILS
# =============================================================================

TRACE_FILE: Optional[TextIO] = None
DEBUG_MODE: bool = False

def tee_print(*args, **kwargs):
    """Print to stdout and the trace file if open."""
    print(*args, **kwargs)
    if TRACE_FILE:
        kwargs_file = kwargs.copy()
        kwargs_file['file'] = TRACE_FILE
        print(*args, **kwargs_file)

def debug_print(*args, **kwargs):
    if DEBUG_MODE:
        tee_print("[DEBUG]", *args, **kwargs)

# =============================================================================
# SEMANTIC TRACER
# =============================================================================

class SemanticTracer:
    def __init__(self, repo_root: Path, roots: Dict[str, call_tree.Root]):
        self.repo_root = repo_root
        self.roots = roots
        self.graph = PythonSemanticGraph()
        self.resolver = HeuristicResolver(self.graph)
        
        # Queue for files we need to ingest
        self.file_queue: Set[Path] = set()
        self.processed_files: Set[Path] = set()
        self.known_missing_targets: Set[str] = set()

    def trace(self, entry_node_id: str):
        # 1. Resolve Entry Point
        debug_print(f"Resolving entry point: {entry_node_id}")
        try:
            ctx, func_node, func_fqn = call_tree.resolve_entry_point(entry_node_id, self.roots, self.repo_root)
        except SystemExit:
            tee_print(f"Could not resolve node: {entry_node_id}")
            return

        debug_print(f"Entry point found in: {ctx.file_path}")
        self.file_queue.add(ctx.file_path)

        # 2. Recursion Loop
        while self.file_queue:
            current_path = self.file_queue.pop()
            if current_path in self.processed_files:
                continue
            
            self._ingest_file(current_path)
            self.processed_files.add(current_path)

            # 3. Heuristic Pass (Iterative)
            # We run this every time we add a file to link new nodes
            self.resolver.resolve_all()
            
            # 4. Discovery Pass (Find new files from unresolved calls)
            self._discover_dependencies()

    def _ingest_file(self, path: Path):
        """Parse a file and add it to the graph."""
        debug_print(f"Ingesting file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            
            # Calculate module name for the graph
            module_name = call_tree.infer_module_name(path, self.roots) or path.stem
            
            file_node = FileNode(path=path, module_name=module_name)
            self.graph.files[path] = file_node
            
            builder = GraphBuilder(self.graph, file_node)
            builder.build(tree)
            
        except Exception as e:
            tee_print(f"Error ingesting {path}: {e}", file=sys.stderr)

    def _discover_dependencies(self):
        """
        Scan the graph for unresolved calls or types and find their source files.
        """
        # A. Check for Missing Types
        checked_types = set()
        for func in self.graph.functions.values():
            if not func.file_path: continue
            
            for var in func.variables.values():
                if var.resolved_type and var.resolved_type not in self.graph.classes:
                    fqn = self.resolver._resolve_type_fqn(var.resolved_type, func.file_path)
                    if fqn and fqn not in checked_types:
                        self._schedule_type_resolution(fqn)
                        checked_types.add(fqn)

        # B. Check for Static Calls
        checked_targets_this_pass = set()

        for func in self.graph.functions.values():
            for call in func.outgoing_calls:
                
                # OPTIMIZATION 1: If we already have the body of this function, skip it!
                if call.resolved_func_id and call.resolved_func_id in self.graph.functions:
                    continue

                # Determine the identifier to check
                target_id = call.resolved_func_id if call.resolved_func_id else call.raw_syntax

                # OPTIMIZATION 2: If we already determined this is external/missing, skip it!
                if target_id in self.known_missing_targets:
                    continue

                # OPTIMIZATION 3: Don't check the same target twice in the same pass
                if target_id in checked_targets_this_pass:
                    continue
                checked_targets_this_pass.add(target_id)

                if call.resolved_func_id:
                    self._schedule_file_for_target(call.resolved_func_id)
                else:
                    self._schedule_file_for_target(call.raw_syntax)
                    
                    if call.caller_var:
                        var = func.variables.get(call.caller_var)
                        if var and var.resolved_type:
                            fqn = self.resolver._resolve_type_fqn(var.resolved_type, func.file_path)
                            if fqn and fqn not in checked_types:
                                self._schedule_type_resolution(fqn)
                                checked_types.add(fqn)

    def _schedule_file_for_target(self, target: str):
        if target in self.known_missing_targets:
            return

        fpath = call_tree.resolve_target_to_file(target, self.roots)
        
        # CASE 1: External Library or Built-in (e.g., torch.tensor)
        if not fpath:
            # debug_print(f"  Target not found (external?): {target}")
            self.known_missing_targets.add(target)
            return

        # CASE 2: New File
        if fpath not in self.processed_files and fpath not in self.file_queue:
            debug_print(f"  Discovered dependency file: {fpath} (via {target})")
            self.file_queue.add(fpath)
            return

        # CASE 3: File already loaded - check for re-exports
        if fpath in self.graph.files:
            file_node = self.graph.files[fpath]
            short_name = target.split(".")[-1]
            
            if short_name in file_node.imports:
                imported_ref = file_node.imports[short_name]
                resolved_fqn = self._resolve_relative_import(imported_ref, file_node.module_name)
                
                if resolved_fqn != target:
                    # Check if the re-export target is already known or missing before scheduling
                    if resolved_fqn in self.graph.functions:
                        return
                    if resolved_fqn in self.known_missing_targets:
                        return

                    debug_print(f"  Following re-export: {target} -> {resolved_fqn}")
                    self._schedule_file_for_target(resolved_fqn)

    def _resolve_relative_import(self, ref: str, current_module: str) -> str:
        """Helper to convert '.sub.func' + 'pkg.mod' -> 'pkg.mod.sub.func'"""
        debug_print(f"  _resolve_relative_import({ref}, {current_module})")
        if not ref.startswith("."):
            return ref
            
        # Count leading dots to determine level
        level = 0
        for char in ref:
            if char == ".": level += 1
            else: break
            
        remainder = ref[level:]
        
        # Split current module to go up 'level - 1' steps
        # level 1 (.) = same directory (no pop)
        # level 2 (..) = up one directory (pop 1)
        parts = current_module.split(".")
        
        # If level is 1, we append to current path. If level > 1, we pop (level-1)
        if level > 1:
            # Pop level-1 parts
            parts = parts[:-(level-1)]
            
        base = ".".join(parts)
        if base:
            return f"{base}.{remainder}"
        return remainder

    def _schedule_type_resolution(self, type_fqn: str):
        """
        Ensures the file defining `type_fqn` is loaded.
        """
        fpath = call_tree.resolve_target_to_file(type_fqn, self.roots)
        if not fpath:
            return

        if fpath not in self.processed_files:
            if fpath not in self.file_queue:
                debug_print(f"  Discovered type definition file: {fpath} (for {type_fqn})")
                self.file_queue.add(fpath)
            return

        if fpath in self.graph.files:
            file_node = self.graph.files[fpath]
            short_name = type_fqn.split(".")[-1]
            
            if short_name in file_node.imports:
                forwarded_fqn = file_node.imports[short_name]
                debug_print(f"  Following re-export: {type_fqn} -> {forwarded_fqn}")
                self._schedule_type_resolution(forwarded_fqn)

    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================

    def print_report(self, pretty: bool = False):
        files_map: Dict[str, List[str]] = {}
        
        for func in self.graph.functions.values():
            if not func.file_path: continue
            
            try:
                f_display = str(func.file_path.relative_to(self.repo_root))
            except:
                f_display = str(func.file_path)
                
            if f_display not in files_map:
                files_map[f_display] = []
            
            block_text = self._format_function(func, pretty)
            files_map[f_display].append(block_text)

        for f_display in sorted(files_map.keys()):
            tee_print(f"{{ FILE: {f_display}")
            for block in files_map[f_display]:
                if pretty:
                    for line in block.splitlines():
                        tee_print(f"  {line}")
                else:
                    tee_print(f"  {block}")
            tee_print("}")

    def _format_function(self, func, pretty: bool) -> str:
        tokens = []
        tokens.append(f"{{ FUNCTION: {func.id}")

        if func.body:
            self._render_logic_block(func.body, tokens, pretty, indent=2)
        
        tokens.append("}")
        
        if pretty:
            return "\n".join(tokens)
        else:
            return " ".join(tokens)

    def _render_logic_block(self, block: LogicBlock, tokens: List[str], pretty: bool, indent: int):
        indent_str = " " * indent if pretty else ""
        
        for instr in block.instructions:
            if isinstance(instr, Assignment):
                continue
            
            elif isinstance(instr, Call):
                if instr.resolved_func_id:
                    if instr.caller_var:
                        class_fqn = instr.resolved_func_id.rpartition(".")[0]
                        text = f"<{class_fqn}>.{instr.func_name}"
                    else:
                        text = instr.resolved_func_id
                else:
                    if instr.caller_var:
                        text = f"<<{instr.caller_var}>>.{instr.func_name}"
                    else:
                        text = instr.raw_syntax

                if instr.children:
                    arg_tokens = []
                    self._render_logic_block(LogicBlock(instr.children), arg_tokens, pretty=False, indent=0)
                    args_str = " ".join(arg_tokens)
                    token = f"{text}( {args_str} )"
                else:
                    token = f"{text}"

                if pretty:
                    tokens.append(f"{indent_str}{token}")
                else:
                    tokens.append(token)
            
            elif isinstance(instr, Keyword):
                if instr.children:
                    arg_tokens = []
                    self._render_logic_block(LogicBlock(instr.children), arg_tokens, pretty=False, indent=0)
                    args_str = " ".join(arg_tokens)
                    text = f"{instr.name}( {args_str} )"
                else:
                    text = instr.name

                if pretty:
                    tokens.append(f"{indent_str}{text}")
                else:
                    tokens.append(text)

            elif isinstance(instr, ControlFlow):
                if not self._has_visible_content(instr.body) and \
                   (not instr.orelse or not self._has_visible_content(instr.orelse)):
                    continue

                if instr.kind == "loop":
                    open_char, close_char = "[", "]"
                else:
                    # Conditional marker {? ... }
                    open_char, close_char = "{?", "}"
                
                if pretty:
                    tokens.append(f"{indent_str}{open_char}")
                    self._render_logic_block(instr.body, tokens, pretty, indent + 2)
                    if instr.orelse:
                         self._render_logic_block(instr.orelse, tokens, pretty, indent + 2)
                    tokens.append(f"{indent_str}{close_char}")
                else:
                    tokens.append(open_char)
                    self._render_logic_block(instr.body, tokens, pretty, 0)
                    if instr.orelse:
                         self._render_logic_block(instr.orelse, tokens, pretty, 0)
                    tokens.append(close_char)

            elif isinstance(instr, TryBlock):
                # NEW: Try/Catch/Finally markers
                
                # 1. Try Block {t
                if pretty:
                    tokens.append(f"{indent_str}{{t")
                    self._render_logic_block(instr.body, tokens, pretty, indent + 2)
                    tokens.append(f"{indent_str}}}")
                else:
                    tokens.append("{t")
                    self._render_logic_block(instr.body, tokens, pretty, 0)
                    tokens.append("}")

                # 2. Handlers {c Type
                for handler in instr.handlers:
                    marker = f"{{c {handler.exception_type}"
                    if pretty:
                        tokens.append(f"{indent_str}{marker}")
                        self._render_logic_block(handler.body, tokens, pretty, indent + 2)
                        tokens.append(f"{indent_str}}}")
                    else:
                        tokens.append(marker)
                        self._render_logic_block(handler.body, tokens, pretty, 0)
                        tokens.append("}")

                # 3. Finally {f
                if instr.finalbody:
                    if pretty:
                        tokens.append(f"{indent_str}{{f")
                        self._render_logic_block(instr.finalbody, tokens, pretty, indent + 2)
                        tokens.append(f"{indent_str}}}")
                    else:
                        tokens.append("{f")
                        self._render_logic_block(instr.finalbody, tokens, pretty, 0)
                        tokens.append("}")

    def _has_visible_content(self, block: LogicBlock) -> bool:
        """Returns True if the block contains any instruction that isn't an Assignment."""
        for instr in block.instructions:
            if not isinstance(instr, Assignment):
                if isinstance(instr, ControlFlow):
                    if self._has_visible_content(instr.body): return True
                    if instr.orelse and self._has_visible_content(instr.orelse): return True
                elif isinstance(instr, TryBlock):
                    # Recurse checks for Try/Catch
                    if self._has_visible_content(instr.body): return True
                    for h in instr.handlers:
                        if self._has_visible_content(h.body): return True
                    if instr.finalbody and self._has_visible_content(instr.finalbody): return True
                else:
                    return True # Call or Keyword
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trace code using Semantic Graph.")
    parser.add_argument("node_id", help="Pytest-style node id (path/to/file.py::Class::method)")
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty-print output")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug

    # Setup Roots
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    if not (repo_root / "src").exists():
        repo_root = script_path.parent
    
    roots = call_tree.build_roots(repo_root)

    # Setup Trace File
    trace_dir = script_path.parent / "trace_history"
    trace_dir.mkdir(exist_ok=True)

    raw_name = args.node_id.replace(os.sep, "_").replace("::", "__")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    trace_path = trace_dir / f"{safe_name}.txt"

    global TRACE_FILE
    try:
        TRACE_FILE = trace_path.open("w", encoding="utf-8")
        
        if args.debug:
            tee_print(f"Starting trace for: {args.node_id}")
            tee_print(f"Repo root: {repo_root}")

        # Run Trace
        tracer = SemanticTracer(repo_root, roots)
        tracer.trace(args.node_id)
        tracer.print_report(pretty=args.pretty)
        
        if args.debug:
            tee_print("Trace complete.")

    finally:
        if TRACE_FILE:
            TRACE_FILE.close()

if __name__ == "__main__":
    main()