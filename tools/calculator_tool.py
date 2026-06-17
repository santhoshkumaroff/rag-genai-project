import ast
import operator
from langchain_core.tools import tool
from models.schemas import ToolResponse

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsafe or unsupported expression")


@tool
async def calculator(expression: str) -> str:
    """Evaluate arithmetic expressions safely"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        response = ToolResponse(
            tool_name="calculator",
            result=str(result),
            status="success",
        )
        return response.result
    except Exception:
        response = ToolResponse(
            tool_name="calculator",
            result="Invalid calculation",
            status="error",
        )
        return response.result
