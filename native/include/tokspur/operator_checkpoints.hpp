#pragma once
#ifdef TOKSPUR_PRIVATE_OPERATOR_CHECKPOINTS
#define TOKSPUR_OPERATOR_CHECKPOINT(graph, category, tag, ...) (graph).operator_checkpoint(category, tag, __VA_ARGS__)
#define TOKSPUR_OPERATOR_CONTEXT(graph, phase, layer, prefix, rows) (graph).set_operator_checkpoint_context(phase, layer, prefix, rows)
#else
#define TOKSPUR_OPERATOR_CHECKPOINT(...) do {} while (false)
#define TOKSPUR_OPERATOR_CONTEXT(...) do {} while (false)
#endif
