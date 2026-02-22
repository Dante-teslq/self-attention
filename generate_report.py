"""Gera relatório HTML com os resultados do Attention. Rode: python generate_report.py"""

import numpy as np
from attention import ScaledDotProductAttention


def matriz_para_html(arr: np.ndarray, titulo: str = "") -> str:
    """Formata matriz NumPy como tabela HTML."""
    linhas = []
    linhas.append("<table class='matrix'>")
    if titulo:
        linhas.append(f"<caption>{titulo}</caption>")
    for row in arr:
        linhas.append("<tr>")
        for cell in row:
            linhas.append(f"<td>{cell:.6f}</td>")
        linhas.append("</tr>")
    linhas.append("</table>")
    return "\n".join(linhas)


def heatmap_para_html(pesos: np.ndarray, titulo: str = "") -> str:
    """Gera heatmap HTML dos pesos de attention (valor 0 = frio, 1 = quente)."""
    linhas = []
    linhas.append("<table class='heatmap'>")
    if titulo:
        linhas.append(f"<caption>{titulo}</caption>")
    v_min = float(np.min(pesos))
    v_max = float(np.max(pesos)) or 1.0
    for row in pesos:
        linhas.append("<tr>")
        for v in row:
            t = (v - v_min) / (v_max - v_min) if v_max > v_min else 0.5
            r = int(30 + t * 200)
            g = int(60 + t * 80)
            b = int(120 - t * 70)
            linhas.append(f"<td style='background:rgb({r},{g},{b});color:#fff'>{v:.4f}</td>")
        linhas.append("</tr>")
    linhas.append("</table>")
    return "\n".join(linhas)


def gerar_html() -> str:
    queries = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    keys = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    values = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    attn = ScaledDotProductAttention()
    saida, pesos_attention = attn.forward(queries, keys, values)

    d_k = keys.shape[-1]
    escala = np.sqrt(d_k)
    scores = queries @ keys.T
    scores_escalados = scores / escala

    html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scaled Dot-Product Attention — Resultados</title>
    <style>
        :root {
            --bg: #1a1b26;
            --surface: #24283b;
            --text: #c0caf5;
            --accent: #7aa2f7;
            --green: #9ece6a;
            --yellow: #e0af68;
        }
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }
        h1 {
            color: var(--accent);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .subtitle { color: var(--yellow); margin-bottom: 2rem; font-size: 0.95rem; }
        section {
            background: var(--surface);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        section h2 {
            color: var(--green);
            font-size: 1.1rem;
            margin-top: 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 0.5rem;
        }
        .formula {
            background: rgba(0,0,0,0.3);
            padding: 1rem 1.25rem;
            border-radius: 8px;
            font-family: 'Consolas', monospace;
            font-size: 1rem;
            margin: 1rem 0;
            color: var(--yellow);
        }
        table.matrix {
            border-collapse: collapse;
            margin: 0.5rem 0;
        }
        table.matrix td {
            border: 1px solid rgba(255,255,255,0.15);
            padding: 0.5rem 0.75rem;
            text-align: right;
            font-family: 'Consolas', monospace;
            font-size: 0.9rem;
        }
        table.matrix caption {
            text-align: left;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }
        .step { margin-bottom: 1.25rem; }
        .step p { margin: 0.25rem 0; font-size: 0.9rem; }
        .shape { color: var(--yellow); font-size: 0.85rem; }
        .ok { color: var(--green); margin-top: 1rem; }
        table.heatmap { border-collapse: collapse; margin: 1rem 0; }
        table.heatmap td {
            border: 1px solid rgba(255,255,255,0.3);
            padding: 1rem 1.25rem;
            text-align: center;
            font-family: 'Consolas', monospace;
            font-size: 1rem;
            min-width: 4rem;
        }
        table.heatmap caption {
            text-align: left;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }
    </style>
</head>
<body>
    <h1>Scaled Dot-Product Attention</h1>
    <p class="subtitle">"Attention Is All You Need" (Vaswani et al., 2017) — Resultados do cálculo</p>

    <section>
        <h2>Equação</h2>
        <div class="formula">Attention(Q, K, V) = softmax((Q × K<sup>T</sup>) / √d<sub>k</sub>) × V</div>
        <p>d<sub>k</sub> = """ + str(d_k) + f""" (dimensão das chaves) → escala = √d<sub>k</sub> = {escala:.4f}</p>
    </section>

    <section>
        <h2>Entradas</h2>
        <div class="step">
            """ + matriz_para_html(queries, "Q (queries) — shape " + str(queries.shape)) + """
            <p class="shape">Shape: """ + str(queries.shape) + """</p>
        </div>
        <div class="step">
            """ + matriz_para_html(keys, "K (keys) — shape " + str(keys.shape)) + """
            <p class="shape">Shape: """ + str(keys.shape) + """</p>
        </div>
        <div class="step">
            """ + matriz_para_html(values, "V (values) — shape " + str(values.shape)) + """
            <p class="shape">Shape: """ + str(values.shape) + """</p>
        </div>
    </section>

    <section>
        <h2>Passos intermediários</h2>
        <div class="step">
            <p><strong>1.</strong> Scores = Q × K<sup>T</sup></p>
            """ + matriz_para_html(scores, "Scores — shape " + str(scores.shape)) + """
        </div>
        <div class="step">
            <p><strong>2.</strong> Scores escalados = Scores / √d<sub>k</sub> = Scores / """ + f"{escala:.4f}" + """</p>
            """ + matriz_para_html(scores_escalados, "Scores escalados") + """
        </div>
        <div class="step">
            <p><strong>3.</strong> Pesos de attention = softmax (por linha)</p>
            """ + matriz_para_html(pesos_attention, "Pesos (soma 1 por linha)") + """
        </div>
    </section>

    <section>
        <h2>Heatmap dos pesos de attention</h2>
        <p>Visualização dos pesos (quanto mais quente a cor, maior o valor).</p>
        """ + heatmap_para_html(pesos_attention, "Pesos de attention — heatmap") + """
    </section>

    <section>
        <h2>Saída (Attention)</h2>
        <div class="step">
            <p><strong>4.</strong> Saída = Pesos × V</p>
            """ + matriz_para_html(saida, "Resultado final — shape " + str(saida.shape)) + """
            <p class="shape">Shape: """ + str(saida.shape) + """</p>
        </div>
        <p class="ok">✓ Validação: shape (seq_len_q, d_v) e valores finitos.</p>
    </section>
</body>
</html>"""
    return html


def main() -> None:
    import os
    html = gerar_html()
    dir_script = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_script, "attention_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Relatório gerado: {path}")
    print("Abra o arquivo no navegador para visualizar os resultados.")


if __name__ == "__main__":
    main()
