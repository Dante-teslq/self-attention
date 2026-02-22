"""
Scaled Dot-Product Attention — "Attention Is All You Need" (Vaswani et al., 2017).

Equação: Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
Implementação em Python com NumPy (classe que executa a equação matricial).
"""

from __future__ import annotations

import numpy as np


class ScaledDotProductAttention:
    """Classe que executa a equação matricial de atenção (scaled dot-product)."""

    @staticmethod
    def _softmax_por_linha(x: np.ndarray) -> np.ndarray:
        """Aplica softmax em cada linha; subtrai o máximo por linha para estabilidade numérica."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        soma_por_linha = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / soma_por_linha

    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Executa a equação de atenção.
        Retorna (saída, pesos_attention) para permitir análise (ex.: heatmap).
        """
        dimensao_chave = keys.shape[-1]
        escala = np.sqrt(dimensao_chave)

        scores = queries @ keys.T
        scores_escalados = scores / escala
        pesos_attention = self._softmax_por_linha(scores_escalados)
        saida = pesos_attention @ values

        return saida, pesos_attention


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Função de conveniência: retorna apenas a saída (compatível com código antigo)."""
    attn = ScaledDotProductAttention()
    saida, _ = attn.forward(queries, keys, values)
    return saida
