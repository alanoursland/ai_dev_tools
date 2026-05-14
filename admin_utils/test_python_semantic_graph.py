import unittest
import ast
from pathlib import Path
from python_semantic_graph import (
    PythonSemanticGraph, 
    GraphBuilder, 
    FileNode, 
    Assignment,
    Call,
    ControlFlow,
    LogicBlock,
    HeuristicResolver,
    ClassNode,
    OriginType
)

class TestMilestone1(unittest.TestCase):
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/test_module.py"), 
            module_name="test_module"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)

    def test_top_level_function(self):
        code = """
def my_func(x: int) -> str:
    pass
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        fqn = "test_module.my_func"
        self.assertIn(fqn, self.graph.functions)
        func = self.graph.functions[fqn]
        self.assertEqual(func.name, "my_func")
        self.assertEqual(func.args["x"], "int")
        self.assertEqual(func.return_type, "str")
        self.assertIn(fqn, self.file_node.functions)

    def test_class_definition(self):
        code = """
class User(BaseModel):
    pass
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        fqn = "test_module.User"
        self.assertIn(fqn, self.graph.classes)
        cls_node = self.graph.classes[fqn]
        self.assertEqual(cls_node.name, "User")
        self.assertEqual(cls_node.bases, ["BaseModel"])

    def test_method_definition(self):
        code = """
class User:
    def save(self):
        pass
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        class_fqn = "test_module.User"
        method_fqn = "test_module.User.save"
        self.assertIn(method_fqn, self.graph.functions)
        cls_node = self.graph.classes[class_fqn]
        self.assertIn("save", cls_node.methods)

    def test_imports(self):
        code = """
import os
from typing import List
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        self.assertIn("os", self.file_node.imports)
        self.assertEqual(self.file_node.imports["List"], "typing.List")


class TestMilestone2(unittest.TestCase):
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/logic_test.py"), 
            module_name="logic_test"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)

    def test_logic_ingestion_assignment(self):
        code = """
def process():
    x = 10
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.process"]
        instr = func.body.instructions[0]
        self.assertIsInstance(instr, Assignment)
        self.assertEqual(instr.targets, ["x"])
        self.assertEqual(instr.value_repr, "10")

    def test_logic_ingestion_call(self):
        code = """
def trigger():
    run_job()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.trigger"]
        instr = func.body.instructions[0]
        self.assertIsInstance(instr, Call)
        self.assertEqual(instr.func_name, "run_job")
        self.assertIsNone(instr.caller_var)

    def test_logic_ingestion_method_call(self):
        code = """
def save_user(u):
    u.save()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.save_user"]
        instr = func.body.instructions[0]
        self.assertIsInstance(instr, Call)
        self.assertEqual(instr.caller_var, "u")
        self.assertEqual(instr.func_name, "save")

    def test_complex_call_name(self):
        code = """
def deep_call():
    self.repo.db.commit()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.deep_call"]
        instr = func.body.instructions[0]
        self.assertEqual(instr.raw_syntax, "self.repo.db.commit")
        self.assertEqual(instr.caller_var, "self.repo.db")
        self.assertEqual(instr.func_name, "commit")

    def test_assignment_type_heuristic(self):
        code = """
def init():
    user = User()
    data = load_json()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.init"]
        assign_1 = func.body.instructions[0]
        self.assertEqual(assign_1.targets, ["user"])
        self.assertEqual(assign_1.inferred_type, "User")
        assign_2 = func.body.instructions[2]
        self.assertEqual(assign_2.targets, ["data"])
        self.assertIsNone(assign_2.inferred_type)

    def test_mixed_assignment_and_call(self):
        code = """
def flow():
    x = foo()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["logic_test.flow"]
        instrs = func.body.instructions
        self.assertEqual(len(instrs), 2)


class TestMilestone3(unittest.TestCase):
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/var_test.py"), 
            module_name="var_test"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)

    def test_argument_registration(self):
        code = """
def handler(ctx: Context, count: int):
    pass
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["var_test.handler"]
        self.assertEqual(func.variables["ctx"].declared_type, "Context")
        self.assertEqual(func.variables["count"].declared_type, "int")

    def test_variable_assignment_history(self):
        code = """
def calc():
    x = 1
    x = 2
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["var_test.calc"]
        var_x = func.variables["x"]
        self.assertEqual(len(var_x.assignments), 2)

    def test_variable_usage_tracking(self):
        code = """
def execute(cmd):
    cmd.run()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["var_test.execute"]
        var_cmd = func.variables["cmd"]
        self.assertEqual(len(var_cmd.usages), 1)

    def test_chained_usage_tracking(self):
        code = """
def render(screen):
    screen.pixels.draw()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        func = self.graph.functions["var_test.render"]
        var_screen = func.variables["screen"]
        self.assertEqual(len(var_screen.usages), 1)
        self.assertEqual(var_screen.usages[0].raw_syntax, "screen.pixels.draw")


class TestMilestone4(unittest.TestCase):
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/res_test.py"), 
            module_name="res_test"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)
        self.resolver = HeuristicResolver(self.graph)

    def test_resolve_from_declaration(self):
        code = """
def strict(x: int):
    pass
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        self.resolver.resolve_all()
        func = self.graph.functions["res_test.strict"]
        self.assertEqual(func.variables["x"].resolved_type, "int")

    def test_resolve_from_constructor_assignment(self):
        code = """
def create():
    u = User()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        self.resolver.resolve_all()
        func = self.graph.functions["res_test.create"]
        self.assertEqual(func.variables["u"].resolved_type, "User")

    def test_resolve_from_constant_assignment(self):
        code = """
def number():
    x = 10
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        self.resolver.resolve_all()
        func = self.graph.functions["res_test.number"]
        self.assertEqual(func.variables["x"].resolved_type, "int")

    def test_declaration_trumps_assignment(self):
        code = """
def mixed():
    x: Base = Derived()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        self.resolver.resolve_all()
        func = self.graph.functions["res_test.mixed"]
        self.assertEqual(func.variables["x"].resolved_type, "Base")

class TestMilestone5(unittest.TestCase):
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/link_test.py"), 
            module_name="link_test"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)
        self.resolver = HeuristicResolver(self.graph)

    def test_link_method_call_internal(self):
        code = """
class User:
    def save(self):
        pass

def trigger():
    u = User()
    u.save()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        user_save_id = "link_test.User.save"
        self.resolver.resolve_all()
        trigger_func = self.graph.functions["link_test.trigger"]
        call_instr = trigger_func.body.instructions[-1]
        self.assertEqual(call_instr.resolved_func_id, user_save_id)

    def test_link_inherited_method(self):
        code = """
class Base:
    def commit(self):
        pass

class Derived(Base):
    pass

def workflow():
    d = Derived()
    d.commit()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        derived_node = self.graph.classes["link_test.Derived"]
        # Explicitly set bases to FQN because simplified test doesn't do cross-file resolution
        # In real scenario, Base would be resolved via imports or same-module logic
        # Here we manually ensure the graph is correct for the test's intent
        # The HeuristicResolver logic relies on the graph having correct FQN refs
        # The GraphBuilder sees "Base" and stores "Base".
        # We need to rely on _resolve_type_fqn doing its job.
        
        base_commit_id = "link_test.Base.commit"
        self.resolver.resolve_all()
        
        workflow_func = self.graph.functions["link_test.workflow"]
        call_instr = workflow_func.body.instructions[-1]
        self.assertEqual(call_instr.resolved_func_id, base_commit_id)

    def test_link_static_call_on_imported_class(self):
        """
        Scenario: Calling a method directly on a Class (static/classmethod context) 
        where the Class is imported. 
        Example: Path.resolve() where Path is imported from pathlib.
        """
        code = """
from pathlib import Path

def check_path():
    Path.resolve()
"""
        tree = ast.parse(code)
        self.builder.build(tree)

        # MOCK THE EXTERNAL WORLD
        # Since we aren't parsing the actual 'pathlib' standard library, 
        # we manually inject the class definition into the graph so the 
        # resolver has something to find.
        path_fqn = "pathlib.Path"
        resolve_fqn = "pathlib.Path.resolve"
        
        path_node = ClassNode(
            id=path_fqn,
            name="Path",
            origin=OriginType.EXTERNAL,
            file_path=None,
            methods={"resolve": resolve_fqn}
        )
        self.graph.classes[path_fqn] = path_node

        # RUN RESOLUTION
        self.resolver.resolve_all()

        # VERIFY
        func = self.graph.functions["link_test.check_path"]
        call_instr = func.body.instructions[0]
        
        # Logic Check:
        # 1. Caller is "Path".
        # 2. "Path" is NOT a local variable.
        # 3. Resolver looks up "Path" in file imports -> "pathlib.Path".
        # 4. Resolver looks up "pathlib.Path" in graph.classes.
        # 5. Resolver finds "resolve" method in that class.
        self.assertEqual(call_instr.resolved_func_id, resolve_fqn)


class TestMilestone6(unittest.TestCase):
    """
    Tests for Control Flow (M6) logic structures.
    """
    def setUp(self):
        self.graph = PythonSemanticGraph()
        self.file_node = FileNode(
            path=Path("src/flow_test.py"), 
            module_name="flow_test"
        )
        self.builder = GraphBuilder(self.graph, self.file_node)

    def test_if_structure(self):
        """
        Scenario: Simple if statement.
        Expected: ControlFlow instruction with correct body instructions.
        """
        code = """
def check(x):
    if x:
        x.save()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        
        func = self.graph.functions["flow_test.check"]
        self.assertEqual(len(func.body.instructions), 1)
        
        if_instr = func.body.instructions[0]
        self.assertIsInstance(if_instr, ControlFlow)
        self.assertEqual(if_instr.kind, "if")
        self.assertEqual(if_instr.condition, "x")
        
        # Check nested instruction
        self.assertEqual(len(if_instr.body.instructions), 1)
        nested_call = if_instr.body.instructions[0]
        self.assertIsInstance(nested_call, Call)
        self.assertEqual(nested_call.func_name, "save")

    def test_loop_structure(self):
        """
        Scenario: For loop iteration.
        """
        code = """
def iterate(items):
    for i in items:
        process(i)
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        
        func = self.graph.functions["flow_test.iterate"]
        loop_instr = func.body.instructions[0]
        
        self.assertIsInstance(loop_instr, ControlFlow)
        self.assertEqual(loop_instr.kind, "loop")
        self.assertEqual(loop_instr.condition, "i in items")
        
        # Check loop body
        nested_call = loop_instr.body.instructions[0]
        self.assertEqual(nested_call.func_name, "process")

    def test_nested_structure(self):
        """
        Scenario: If inside a Loop.
        """
        code = """
def complex_flow(items):
    for i in items:
        if i.valid:
            i.save()
"""
        tree = ast.parse(code)
        self.builder.build(tree)
        
        func = self.graph.functions["flow_test.complex_flow"]
        
        # Level 1: Loop
        loop = func.body.instructions[0]
        self.assertIsInstance(loop, ControlFlow)
        
        # Level 2: If
        if_stmt = loop.body.instructions[0]
        self.assertIsInstance(if_stmt, ControlFlow)
        self.assertEqual(if_stmt.kind, "if")
        
        # Level 3: Call
        call = if_stmt.body.instructions[0]
        self.assertIsInstance(call, Call)
        self.assertEqual(call.func_name, "save")

if __name__ == '__main__':
    unittest.main()