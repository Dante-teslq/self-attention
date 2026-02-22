"""Teste do Scaled Dot-Product Attention: exemplo com Q, K, V e validação da saída."""

import numpy as np
from attention import ScaledDotProductAttention


def main() -> None:
    # Dados de exemplo: Q e K (2, 3), V (2, 2)
    queries = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    keys = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    values = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    print("=== Entradas ===\n")
    print("Q (queries), shape", queries.shape, ":")
    print(queries)
    print("\nK (keys), shape", keys.shape, ":")
    print(keys)
    print("\nV (values), shape", values.shape, ":")
    print(values)

    attn = ScaledDotProductAttention()
    saida, _ = attn.forward(queries, keys, values)

    print("\n=== Saída (Attention) ===\n")
    print("Shape:", saida.shape)
    print(saida)

    assert saida.shape == (queries.shape[0], values.shape[1]), (
        f"Shape esperada (seq_len_q, d_v) = ({queries.shape[0]}, {values.shape[1]}), "
        f"obtida {saida.shape}"
    )
    assert np.all(np.isfinite(saida)), "Saída deve conter apenas valores finitos."
    print("\n[OK] Validação passou: shape e valores finitos corretos.")


if __name__ == "__main__":
    main()
