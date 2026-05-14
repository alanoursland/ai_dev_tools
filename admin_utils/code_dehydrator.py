import ast
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. DATA STRUCTURES
# =============================================================================

@dataclass
class DehydrationContext:
    """
    Holds the static analysis context for a specific module/file.
    """
    file_display: str               # e.g. "src/kinopulse/core/state.py"
    module_name: Optional[str]      # e.g. "kinopulse.core.state"
    import_map: Dict[str, str]      # e.g. "np" -> "numpy"
    local_defs: Dict[str, str]      # e.g. "Helper" -> "kinopulse.core.state.Helper"


@dataclass
class DehydratedBlock:
    """
    The result of dehydrating a single function or method.
    """
    name: str                       # Qualified name: module.Class.method
    content: str                    # The DSL formatted string
    calls: List[Tuple[str, Any]]    # The raw call structure list


# =============================================================================
# 2. NAME RESOLUTION & AST HELPERS
# =============================================================================

def parse_module(file_path: Path) -> Optional[ast.Module]:
    """
    Reads a file and returns its AST. Returns None on failure.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        print(f"Syntax error parsing {file_path}: {e}", file=sys.stderr)
        return None

    return tree


def find_ast_node(target: str, tree: ast.Module) -> Optional[ast.AST]:
    """
    Finds a FunctionDef or ClassDef node within a tree based on a dotted target string.
    Supports 'func_name' or 'Class.method_name'.
    """
    parts = target.split(".")
    name = parts[-1]
    parent = parts[-2] if len(parts) > 1 else None

    # Direct function / class at module level
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node

    # Method inside class
    if parent:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == parent:
                for n in node.body:
                    if isinstance(n, ast.FunctionDef) and n.name == name:
                        return n

    return None


def get_test_function_node(
    tree: ast.Module, cls_name: Optional[str], func_name: str
) -> Optional[ast.FunctionDef]:
    """
    Specific lookup for test entry points (handling optional class containers).
    """
    if cls_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and n.name == func_name:
                    return n
            return None

    return None


def get_raw_call_name(func: ast.expr) -> Optional[str]:
    """Extract a raw name from a call target, like 'foo', 'torch.mean', 'obj.method'."""
    if isinstance(func, ast.Name):
        return func.id

    if isinstance(func, ast.Attribute):
        parts: List[str] = []
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value  # type: ignore
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None
    
    # Handle Subscripts (e.g. BaseClass[T])
    if isinstance(func, ast.Subscript):
        return get_raw_call_name(func.value)

    return None


def qualify_name(
    name: str,
    ctx: DehydrationContext,
    var_types: Dict[str, str],
    self_prefix: Optional[str] = None,
) -> str:
    """
    Qualify a raw call name using imports, local definitions, variable types, and self context.
    """
    # 1. Handle "self."
    if name.startswith("self."):
        parts = name.split(".")
        if len(parts) == 2:
            # self.method
            if self_prefix:
                return f"{self_prefix}.{parts[1]}"
            return name
            
        if len(parts) >= 3:
            # self.attr.method -> self.<<attr>>.method
            base = parts[0]
            attr = parts[1]
            rest = ".".join(parts[2:])
            return f"{base}.<<{attr}>>.{rest}"
        return name

    # 3. Dotted name resolution
    if "." in name:
        head, rest = name.split(".", 1)

        # Instance call? (head is a variable with known type)
        if head in var_types:
            base = var_types[head]
            return f"<{base}>.{rest}"

        # Imported alias?
        if head in ctx.import_map:
            base = ctx.import_map[head]
            return f"{base}.{rest}" if rest else base

        # Local class/func?
        if head in ctx.local_defs:
            base = ctx.local_defs[head]
            return f"{base}.{rest}" if rest else base

        return f"<<{head}>>.{rest}"

    # 4. Simple name resolution
    if name in ctx.import_map:
        return ctx.import_map[name]

    if name in ctx.local_defs:
        return ctx.local_defs[name]

    return name


def build_import_map(tree: ast.Module, module_name: Optional[str] = None) -> Dict[str, str]:
    """
    Build a map of local names -> fully qualified names based on import statements.
    """
    import_map: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname if alias.asname else alias.name.split(".")[0]
                import_map[key] = alias.name

        elif isinstance(node, ast.ImportFrom):
            # Resolve module path
            if node.level and node.level > 0:
                if not module_name:
                    continue 
                
                parts = module_name.split(".")
                parent_parts = parts[:-node.level] if node.level <= len(parts) else []
                base_pkg = ".".join(parent_parts)
                
                module_suffix = node.module or ""
                resolved_module = f"{base_pkg}.{module_suffix}" if base_pkg and module_suffix else (base_pkg or module_suffix)
            else:
                resolved_module = node.module or ""

            for alias in node.names:
                local_name = alias.asname or alias.name
                if resolved_module:
                    fq = f"{resolved_module}.{alias.name}"
                else:
                    fq = alias.name
                import_map[local_name] = fq

    return import_map


def build_local_defs(tree: ast.Module, module_name: Optional[str]) -> Dict[str, str]:
    """
    Map top-level definitions to their qualified names.
    """
    local_defs: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if module_name:
                local_defs[node.name] = f"{module_name}.{node.name}"
            else:
                local_defs[node.name] = node.name
    return local_defs


# =============================================================================
# 3. VISITORS
# =============================================================================

class VarTypeCollector(ast.NodeVisitor):
    def __init__(self, ctx: DehydrationContext, self_prefix: Optional[str] = None):
        self.ctx = ctx
        self.self_prefix = self_prefix
        self.var_types: Dict[str, str] = {}

    def _set_container_type(self, target: ast.expr, typename: str) -> None:
        if isinstance(target, ast.Name):
            self.var_types[target.id] = typename

    def visit_Assign(self, node: ast.Assign) -> Any:
        value = node.value
        
        if isinstance(value, ast.List):
            for t in node.targets: self._set_container_type(t, "list")
        elif isinstance(value, ast.Dict):
            for t in node.targets: self._set_container_type(t, "dict")
        elif isinstance(value, ast.Set):
            for t in node.targets: self._set_container_type(t, "set")
        
        elif isinstance(value, ast.Call):
            raw = get_raw_call_name(value.func)
            if raw in {"list", "dict", "set"}:
                for t in node.targets: self._set_container_type(t, raw)
            elif raw:
                qual = qualify_name(raw, self.ctx, self.var_types, self.self_prefix)
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.var_types[t.id] = qual
        
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        target = node.target
        value = node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            raw = get_raw_call_name(value.func)
            if raw:
                qual = qualify_name(raw, self.ctx, self.var_types, self.self_prefix)
                self.var_types[target.id] = qual
        self.generic_visit(node)


class CallCollector(ast.NodeVisitor):
    def __init__(self, ctx: DehydrationContext, var_types: Dict[str, str], self_prefix: Optional[str] = None):
        self.stack: List[List[Tuple[str, Any]]] = [[]]
        self.ctx = ctx
        self.var_types = var_types
        self.self_prefix = self_prefix

    @property
    def current(self) -> List[Tuple[str, Any]]:
        return self.stack[-1]

    def _push_block(self, kind: str, body_nodes: List[ast.AST], else_nodes: List[ast.AST] = []) -> None:
        group: List[Tuple[str, Any]] = []
        self.stack.append(group)
        for stmt in body_nodes:
            self.visit(stmt)
        for stmt in else_nodes:
            self.visit(stmt)
        self.stack.pop()
        if group:
            self.current.append((kind, group))

    def visit_For(self, node: ast.For) -> Any:
        self._push_block("loop", node.body, node.orelse)

    def visit_While(self, node: ast.While) -> Any:
        self._push_block("loop", node.body, node.orelse)

    def visit_If(self, node: ast.If) -> Any:
        self._push_block("cond", node.body, node.orelse)

    def visit_Call(self, node: ast.Call) -> Any:
        raw = get_raw_call_name(node.func)
        if raw:
            qualified = qualify_name(raw, self.ctx, self.var_types, self.self_prefix)
            self.current.append(("call", qualified))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        self.generic_visit(node)
        self.current.append(("keyword", "return"))

    def visit_Raise(self, node: ast.Raise) -> Any:
        self.generic_visit(node)
        self.current.append(("keyword", "raise"))
    
    def visit_Break(self, node: ast.Break) -> Any:
        self.current.append(("keyword", "break"))

    def visit_Continue(self, node: ast.Continue) -> Any:
        self.current.append(("keyword", "continue"))


# =============================================================================
# 4. FORMATTING
# =============================================================================

class Formatter:
    @staticmethod
    def format_block(function_name: str, calls_structure: List[Tuple[str, Any]], pretty: bool = False) -> str:
        
        if pretty:
            dsl_lines = [f"{{ FUNCTION: {function_name}"]
            
            def emit_pretty(items: List[Tuple[str, Any]], indent: str = ""):
                for kind, payload in items:
                    if kind == "call":
                        dsl_lines.append(f"{indent}{payload}")
                    elif kind == "keyword":
                        dsl_lines.append(f"{indent}( {payload} )")
                    elif kind == "loop":
                        dsl_lines.append(f"{indent}[")
                        emit_pretty(payload, indent + "  ")
                        dsl_lines.append(f"{indent}]")
                    elif kind == "cond":
                        dsl_lines.append(f"{indent}(")
                        emit_pretty(payload, indent + "  ")
                        dsl_lines.append(f"{indent})")
                        
            emit_pretty(calls_structure, "  ")
            dsl_lines.append("}")
            return "\n".join(dsl_lines)
            
        else:
            # Compact Mode (Single line per function)
            tokens = [f"{{ FUNCTION: {function_name}"]
            
            def emit_compact(items: List[Tuple[str, Any]]):
                for kind, payload in items:
                    if kind == "call":
                        tokens.append(str(payload))
                    elif kind == "keyword":
                        tokens.append(f"( {payload} )")
                    elif kind == "loop":
                        tokens.append("[")
                        emit_compact(payload)
                        tokens.append("]")
                    elif kind == "cond":
                        tokens.append("(")
                        emit_compact(payload)
                        tokens.append(")")
            
            emit_compact(calls_structure)
            tokens.append("}")
            return " ".join(tokens)


# =============================================================================
# 5. MAIN DEHYDRATOR CLASS
# =============================================================================

class Dehydrator:
    """
    Main entry point for dehydrating code.
    """
    
    @staticmethod
    def create_context(file_path: str, source_code: str, module_name: Optional[str] = None) -> DehydrationContext:
        tree = ast.parse(source_code, filename=file_path)
        import_map = build_import_map(tree, module_name)
        local_defs = build_local_defs(tree, module_name)
        
        return DehydrationContext(
            file_display=file_path,
            module_name=module_name,
            import_map=import_map,
            local_defs=local_defs
        )

    def dehydrate_node(self, node: ast.AST, ctx: DehydrationContext, parent_name: Optional[str] = None, pretty: bool = False) -> List[DehydratedBlock]:
        """
        Process an AST node (ClassDef or FunctionDef) and return DehydratedBlocks.
        """
        results: List[DehydratedBlock] = []

        if isinstance(node, ast.ClassDef):
            class_fqn = f"{ctx.module_name}.{node.name}" if ctx.module_name else node.name
            
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    method_name = f"{class_fqn}.{n.name}"
                    self_prefix = class_fqn
                    block = self._process_function(n, ctx, method_name, self_prefix, pretty)
                    results.append(block)

        elif isinstance(node, ast.FunctionDef):
            if parent_name:
                func_name = parent_name
                self_prefix = None
            else:
                func_name = f"{ctx.module_name}.{node.name}" if ctx.module_name else node.name
                self_prefix = None

            block = self._process_function(node, ctx, func_name, self_prefix, pretty)
            results.append(block)

        return results

    def _process_function(self, node: ast.FunctionDef, ctx: DehydrationContext, func_name: str, self_prefix: Optional[str], pretty: bool) -> DehydratedBlock:
        var_collector = VarTypeCollector(ctx, self_prefix)
        for stmt in node.body:
            var_collector.visit(stmt)
        
        call_collector = CallCollector(ctx, var_collector.var_types, self_prefix)
        for stmt in node.body:
            call_collector.visit(stmt)
        
        formatted = Formatter.format_block(func_name, call_collector.current, pretty=pretty)
        
        return DehydratedBlock(
            name=func_name,
            content=formatted,
            calls=call_collector.current
        )

    def dehydrate_module(self, source_code: str, file_display: str, module_name: Optional[str] = None, pretty: bool = False) -> List[DehydratedBlock]:
        """
        Parse an entire module string and return blocks for every top-level function and class method.
        """
        tree = ast.parse(source_code)
        ctx = DehydrationContext(
            file_display=file_display,
            module_name=module_name,
            import_map=build_import_map(tree, module_name),
            local_defs=build_local_defs(tree, module_name)
        )
        
        all_blocks: List[DehydratedBlock] = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                blocks = self.dehydrate_node(node, ctx, pretty=pretty)
                all_blocks.extend(blocks)
                
        return all_blocks