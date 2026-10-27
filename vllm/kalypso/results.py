"""Shared semantic query result formatting, without engine dependencies."""


def format_results(out_ctxs) -> list[dict]:
    results = []
    for ctx in out_ctxs:
        outputs = []
        for item in ctx.output:
            if isinstance(item, dict) and len(item) == 1:
                op_name, value = next(iter(item.items()))
                outputs.append({"op": op_name, "value": value})
            else:
                outputs.append({"op": "unknown", "value": item})
        results.append(
            {
                "idx": getattr(ctx.state, "idx", None),
                "worker": getattr(getattr(ctx.state, "worker", None), "name", None),
                "input_context": getattr(ctx.input, "data", None),
                "right_input_context": getattr(ctx.input, "right_data", None),
                "outputs": outputs,
            }
        )
    return results
