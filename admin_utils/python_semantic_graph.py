import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from enum import Enum

# =============================================================================
# 1. CORE ENUMS & TYPES
# =============================================================================

class OriginType(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

# =============================================================================
# 2. THE INSTRUCTION SET (Recursive)
# =============================================================================

class Instruction:
    lineno: int = 0

@dataclass
class Assignment(Instruction):
    targets: List[str]
    value_repr: str
    inferred_type: Optional[str] = None
    derived_from_var: Optional[str] = None

@dataclass
class Call(Instruction):
    raw_syntax: str
    func_name: str
    caller_var: Optional[str]
    arguments: List[str]
    resolved_func_id: Optional[str] = None
    children: List[Instruction] = field(default_factory=list)

@dataclass
class Keyword(Instruction):
    name: str
    # Support children for return/raise values
    children: List[Instruction] = field(default_factory=list) 

@dataclass
class LogicBlock:
    """A container for a sequence of instructions."""
    instructions: List[Instruction] = field(default_factory=list)

@dataclass
class ControlFlow(Instruction):
    """Represents branching (if) or looping (for/while) logic."""
    kind: str                     # "if", "loop"
    condition: str
    body: LogicBlock
    orelse: Optional[LogicBlock] = None

@dataclass
class CatchBlock:
    """Represents a specific except block."""
    exception_type: str
    body: LogicBlock

@dataclass
class TryBlock(Instruction):
    """Represents try/except/finally structures."""
    body: LogicBlock
    handlers: List[CatchBlock] = field(default_factory=list)
    finalbody: Optional[LogicBlock] = None

# =============================================================================
# 3. THE GRAPH NODES
# =============================================================================

@dataclass
class VariableNode:
    name: str
    declared_type: Optional[str] = None
    resolved_type: Optional[str] = None
    assignments: List[Assignment] = field(default_factory=list)
    usages: List[Call] = field(default_factory=list)

@dataclass
class FunctionNode:
    id: str
    name: str
    origin: OriginType
    file_path: Optional[Path]
    
    args: Dict[str, str] = field(default_factory=dict)
    return_type: Optional[str] = None
    
    variables: Dict[str, VariableNode] = field(default_factory=dict)
    body: Optional[LogicBlock] = None
    
    outgoing_calls: List[Call] = field(default_factory=list)

@dataclass
class ClassNode:
    id: str
    name: str
    origin: OriginType
    file_path: Optional[Path]
    
    bases: List[str] = field(default_factory=list)
    methods: Dict[str, str] = field(default_factory=dict)

@dataclass
class FileNode:
    path: Path
    module_name: str
    imports: Dict[str, str] = field(default_factory=dict)
    
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)

# =============================================================================
# 4. THE STORE
# =============================================================================

class PythonSemanticGraph:
    def __init__(self):
        self.files: Dict[Path, FileNode] = {}
        self.classes: Dict[str, ClassNode] = {}
        self.functions: Dict[str, FunctionNode] = {}
        
    def get_function(self, fqn: str) -> Optional[FunctionNode]:
        return self.functions.get(fqn)

# =============================================================================
# 5. THE BUILDER
# =============================================================================

class GraphBuilder(ast.NodeVisitor):
    def __init__(self, graph: PythonSemanticGraph, context: FileNode):
        self.graph = graph
        self.context = context
        
        self.scope_stack: List[str] = [context.module_name]
        self.current_class: Optional[ClassNode] = None
        self.current_function: Optional[FunctionNode] = None
        
        # Block Stack for nesting
        self.block_stack: List[LogicBlock] = []

    @property
    def current_block(self) -> Optional[LogicBlock]:
        return self.block_stack[-1] if self.block_stack else None

    def build(self, tree: ast.Module):
        self.graph.files[self.context.path] = self.context
        self.visit(tree)

    def _get_current_fqn_prefix(self) -> str:
        return ".".join(self.scope_stack)

    # --- Scope Visitors ---

    def visit_ClassDef(self, node: ast.ClassDef):
        class_fqn = f"{self._get_current_fqn_prefix()}.{node.name}"
        
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except:
                bases.append("<?>")

        class_node = ClassNode(
            id=class_fqn,
            name=node.name,
            origin=OriginType.INTERNAL,
            file_path=self.context.path,
            bases=bases
        )
        
        self.graph.classes[class_fqn] = class_node
        self.context.classes.append(class_fqn)
        
        self.scope_stack.append(node.name)
        prev_class = self.current_class
        self.current_class = class_node
        
        self.generic_visit(node)
        
        self.current_class = prev_class
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_fqn = f"{self._get_current_fqn_prefix()}.{node.name}"
        
        args = self._extract_args(node)
        ret_type = self._extract_annotation(node.returns)
        
        func_node = FunctionNode(
            id=func_fqn,
            name=node.name,
            origin=OriginType.INTERNAL,
            file_path=self.context.path,
            args=args,
            return_type=ret_type
        )
        
        self.graph.functions[func_fqn] = func_node
        
        if self.current_class:
            self.current_class.methods[node.name] = func_fqn
        else:
            self.context.functions.append(func_fqn)
            
        self.scope_stack.append(node.name)
        prev_func = self.current_function
        
        self.current_function = func_node
        
        root_block = LogicBlock()
        func_node.body = root_block
        self.block_stack.append(root_block)
        
        for arg_name, arg_type in args.items():
            var_node = self._get_or_create_var(arg_name)
            var_node.declared_type = arg_type if arg_type != "Any" else None

        for stmt in node.body:
            self.visit(stmt)
        
        self.block_stack.pop() # Pop Root Block
        self.current_function = prev_func
        self.scope_stack.pop()

    # --- Logic Visitors ---

    def visit_Assign(self, node: ast.Assign):
        if not self.current_block: return
        
        targets = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.append(t.id)
            elif isinstance(t, ast.Attribute):
                try: targets.append(ast.unparse(t))
                except: pass
        
        if not targets: return

        try: val_repr = ast.unparse(node.value)
        except: val_repr = "<complex>"

        instr = Assignment(targets=targets, value_repr=val_repr)
        instr.lineno = node.lineno
        self.current_block.instructions.append(instr)
        
        raw_call_name = None
        if isinstance(node.value, ast.Call):
            raw_call_name = self._extract_call_name(node.value)
        elif isinstance(node.value, ast.Attribute):
            raw_call_name = self._extract_call_name(node.value)
            
        if raw_call_name and "." in raw_call_name:
             instr.derived_from_var = raw_call_name.rsplit(".", 1)[0]
        
        for t in targets:
            if "." not in t:
                var = self._get_or_create_var(t)
                var.assignments.append(instr)

        if raw_call_name:
             last_part = raw_call_name.split(".")[-1]
             if last_part and last_part[0].isupper():
                 instr.inferred_type = raw_call_name
        elif isinstance(node.value, ast.Constant):
             if isinstance(node.value.value, str):
                 instr.inferred_type = "str"
             elif isinstance(node.value.value, int):
                 instr.inferred_type = "int"
        
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if not self.current_block: return
        
        target_name = None
        if isinstance(node.target, ast.Name):
            target_name = node.target.id
        elif isinstance(node.target, ast.Attribute):
            try: target_name = ast.unparse(node.target)
            except: pass
        
        if not target_name: return

        if "." not in target_name:
            var = self._get_or_create_var(target_name)
            try: var.declared_type = ast.unparse(node.annotation)
            except: pass

        if node.value:
            try: val_repr = ast.unparse(node.value)
            except: val_repr = "<complex>"

            instr = Assignment(targets=[target_name], value_repr=val_repr)
            instr.lineno = node.lineno
            self.current_block.instructions.append(instr)

            if "." not in target_name:
                var = self._get_or_create_var(target_name)
                var.assignments.append(instr)
            
            if isinstance(node.value, ast.Call):
                raw_name = self._extract_call_name(node.value)
                if raw_name:
                    if "." in raw_name:
                         instr.derived_from_var = raw_name.rsplit(".", 1)[0]
                    last_part = raw_name.split(".")[-1]
                    if last_part and last_part[0].isupper():
                        instr.inferred_type = raw_name
            elif isinstance(node.value, ast.Constant):
                 if isinstance(node.value.value, str):
                     instr.inferred_type = "str"
                 elif isinstance(node.value.value, int):
                     instr.inferred_type = "int"
        
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not self.current_block: return
        
        raw_name = self._extract_call_name(node)
        if not raw_name: return

        if "." in raw_name:
            caller, method = raw_name.rsplit(".", 1)
        else:
            caller, method = None, raw_name

        args_repr = []
        for a in node.args:
            try: args_repr.append(ast.unparse(a))
            except: args_repr.append("<?>")

        instr = Call(
            raw_syntax=raw_name,
            func_name=method,
            caller_var=caller,
            arguments=args_repr
        )
        instr.lineno = node.lineno
        
        # 1. Visit Caller logic
        self.visit(node.func)

        # 2. Visit Arguments (Nested)
        args_block = LogicBlock()
        self.block_stack.append(args_block)
        
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword)
            
        self.block_stack.pop()
        instr.children = args_block.instructions

        # 3. Append Call
        self.current_block.instructions.append(instr)
        
        if caller_var_simple := caller: 
             if "." not in caller:
                 var = self._get_or_create_var(caller)
                 var.usages.append(instr)
             else:
                 root = caller.split(".")[0]
                 var = self._get_or_create_var(root)
                 var.usages.append(instr)

        if self.current_function:
            self.current_function.outgoing_calls.append(instr)

    # --- Control Flow Visitors ---

    def _handle_control_flow(self, kind: str, condition: str, body: List[ast.stmt], orelse: List[ast.stmt] = []):
        if not self.current_block: return

        cf_instr = ControlFlow(kind=kind, condition=condition, body=LogicBlock())
        self.current_block.instructions.append(cf_instr)

        self.block_stack.append(cf_instr.body)
        for stmt in body:
            self.visit(stmt)
        self.block_stack.pop()

        if orelse:
            cf_instr.orelse = LogicBlock()
            self.block_stack.append(cf_instr.orelse)
            for stmt in orelse:
                self.visit(stmt)
            self.block_stack.pop()

    def visit_If(self, node: ast.If):
        try: cond = ast.unparse(node.test)
        except: cond = "?"
        self._handle_control_flow("if", cond, node.body, node.orelse)

    def visit_For(self, node: ast.For):
        try: target = ast.unparse(node.target)
        except: target = "?"
        try: iter_ = ast.unparse(node.iter)
        except: iter_ = "?"
        cond = f"{target} in {iter_}"
        self._handle_control_flow("loop", cond, node.body, node.orelse)

    def visit_While(self, node: ast.While):
        try: cond = ast.unparse(node.test)
        except: cond = "?"
        self._handle_control_flow("loop", cond, node.body, node.orelse)

    def visit_Try(self, node: ast.Try):
        # CASE 1: Module Level (No active logic block)
        # We must still visit children to find Imports, Classes, or Functions 
        # defined inside the try block.
        if not self.current_block:
            self.generic_visit(node)
            return

        # CASE 2: Function Level (Active logic block)
        # We capture the control flow structure.
        
        # 1. Try Body
        try_body = LogicBlock()
        self.block_stack.append(try_body)
        for stmt in node.body:
            self.visit(stmt)
        self.block_stack.pop()

        # 2. Handlers
        handlers = []
        for handler in node.handlers:
            catch_body = LogicBlock()
            self.block_stack.append(catch_body)
            for stmt in handler.body:
                self.visit(stmt)
            self.block_stack.pop()
            
            exc_name = "Exception"
            if handler.type:
                try: exc_name = ast.unparse(handler.type)
                except: pass
            
            handlers.append(CatchBlock(exception_type=exc_name, body=catch_body))

        # 3. Finally
        final_body = None
        if node.finalbody:
            final_body = LogicBlock()
            self.block_stack.append(final_body)
            for stmt in node.finalbody:
                self.visit(stmt)
            self.block_stack.pop()

        instr = TryBlock(body=try_body, handlers=handlers, finalbody=final_body)
        instr.lineno = node.lineno
        self.current_block.instructions.append(instr)
    def visit_Return(self, node: ast.Return):
        if not self.current_block: return
        
        args_block = LogicBlock()
        self.block_stack.append(args_block)
        if node.value:
            self.visit(node.value)
        self.block_stack.pop()
        
        self.current_block.instructions.append(
            Keyword(name="return", children=args_block.instructions)
        )

    def visit_Break(self, node: ast.Break):
        if not self.current_block: return
        self.current_block.instructions.append(Keyword(name="break"))

    def visit_Continue(self, node: ast.Continue):
        if not self.current_block: return
        self.current_block.instructions.append(Keyword(name="continue"))

    def visit_Raise(self, node: ast.Raise):
        if not self.current_block: return
        
        args_block = LogicBlock()
        self.block_stack.append(args_block)
        if node.exc:
            self.visit(node.exc)
        if node.cause:
            self.visit(node.cause)
        self.block_stack.pop()
        
        self.current_block.instructions.append(
            Keyword(name="raise", children=args_block.instructions)
        )

    # --- Helpers ---

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            alias = name.asname or name.name
            self.context.imports[alias] = name.name

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # Calculate the module path prefix based on relative level (dots)
        # level 0 = absolute, 1 = ., 2 = .., etc.
        dots = "." * node.level
        module_part = node.module if node.module else ""
        
        # If it's a relative import, keep the dots so the resolver knows
        if node.level > 0:
            source_module = f"{dots}{module_part}"
        else:
            source_module = module_part

        for name in node.names:
            alias = name.asname or name.name
            # Store the source: e.g. ".mass_matrix.compute_constraint_jacobian"
            fqn = f"{source_module}.{name.name}" if source_module else name.name
            self.context.imports[alias] = fqn

    def _get_or_create_var(self, name: str) -> VariableNode:
        if not self.current_function:
            raise ValueError("Cannot access variables outside function scope")
        if name not in self.current_function.variables:
            self.current_function.variables[name] = VariableNode(name=name)
        return self.current_function.variables[name]

    def _extract_args(self, node: ast.FunctionDef) -> Dict[str, str]:
        args = {}
        for arg in node.args.args:
            t_hint = self._extract_annotation(arg.annotation) if arg.annotation else None
            if t_hint:
                args[arg.arg] = t_hint
            else:
                args[arg.arg] = "Any"
        return args

    def _extract_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        if node is None: return None
        try: return ast.unparse(node)
        except: return "<complex>"

    def _extract_call_name(self, node: Union[ast.Call, ast.expr]) -> Optional[str]:
        if isinstance(node, ast.Call):
            return self._extract_call_name(node.func)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._extract_call_name(node.value)
            if val: return f"{val}.{node.attr}"
        return None

# =============================================================================
# 6. THE HEURISTIC RESOLVER
# =============================================================================

class HeuristicResolver:
    def __init__(self, graph: PythonSemanticGraph):
        self.graph = graph

    def resolve_all(self):
        self.pass_1_type_propagation()
        self.pass_2_call_linking()

    def pass_1_type_propagation(self):
        changed = True
        while changed:
            changed = False
            for func_node in self.graph.functions.values():
                for var_node in func_node.variables.values():
                    if var_node.resolved_type: continue
                    
                    if var_node.declared_type:
                        var_node.resolved_type = var_node.declared_type
                        changed = True
                        continue
                        
                    for assignment in reversed(var_node.assignments):
                        if assignment.inferred_type:
                            var_node.resolved_type = assignment.inferred_type
                            changed = True
                            break
                        
                        if assignment.derived_from_var:
                            source_var = func_node.variables.get(assignment.derived_from_var)
                            if source_var and source_var.resolved_type:
                                var_node.resolved_type = source_var.resolved_type
                                changed = True
                                break

    def pass_2_call_linking(self):
        for func_node in self.graph.functions.values():
            for call in func_node.outgoing_calls:
                
                if not call.caller_var:
                     fqn = self._resolve_type_fqn(call.func_name, func_node.file_path)
                     if fqn:
                         call.resolved_func_id = fqn
                     continue

                if "." in call.caller_var:
                    continue
                
                class_fqn = None
                
                var_node = func_node.variables.get(call.caller_var)
                if var_node and var_node.resolved_type:
                    class_fqn = self._resolve_type_fqn(var_node.resolved_type, func_node.file_path)
                
                if not class_fqn:
                    class_fqn = self._resolve_type_fqn(call.caller_var, func_node.file_path)

                if not class_fqn: continue
                    
                class_node = self.graph.classes.get(class_fqn)
                if class_node:
                    method_id = self._resolve_method(class_node, call.func_name)
                    if method_id:
                        call.resolved_func_id = method_id
                else:
                    call.resolved_func_id = f"{class_fqn}.{call.func_name}"

    def _resolve_type_fqn(self, type_name: str, file_path: Optional[Path]) -> Optional[str]:
        if type_name in self.graph.classes:
            return type_name
        if not file_path or file_path not in self.graph.files:
            return None
        file_node = self.graph.files[file_path]
        if type_name in file_node.imports:
            return file_node.imports[type_name]
        candidate = f"{file_node.module_name}.{type_name}"
        if candidate in self.graph.classes:
            return candidate
        return None

    def _resolve_method(self, class_node: ClassNode, method_name: str) -> Optional[str]:
        if method_name in class_node.methods:
            return class_node.methods[method_name]
        for base_name in class_node.bases:
            base_fqn = self._resolve_type_fqn(base_name, class_node.file_path)
            if not base_fqn:
                if base_name in self.graph.classes: base_fqn = base_name
                else: continue
            base_node = self.graph.classes.get(base_fqn)
            if base_node:
                res = self._resolve_method(base_node, method_name)
                if res: return res
        return None