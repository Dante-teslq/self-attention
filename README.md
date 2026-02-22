# Implementação do Mecanismo de Self-Attention

Este repositório apresenta uma implementação do **Scaled Dot-Product Attention**, conforme descrito no paper **“Attention Is All You Need”**, utilizando **Python** e **NumPy**, sem o uso de bibliotecas de alto nível de Deep Learning.

O objetivo do projeto é compreender o funcionamento do mecanismo de Self-Attention por meio da manipulação das matrizes **Query (Q)**, **Key (K)** e **Value (V)**.

---

## 📌 Requisitos

- Python 3.7 ou superior
- NumPy

Instalação da dependência:

```bash
pip install numpy
```

# ▶ Instruções de como rodar o código

Clone o repositório:
```
git clone https://github.com/GLagess/Implementa-o-do-Mecanismo-de-Self-Attention.git
```
Acesse a pasta do projeto:
```
cd Implementa-o-do-Mecanismo-de-Self-Attention
```
Execute o script de teste:
```
python test_attention.py
```

# 🧠 Explicação da normalização (Scaling Factor √dₖ)

Após o cálculo do produto escalar entre as matrizes Q × Kᵀ, o resultado é dividido pela raiz quadrada da dimensão das chaves (√dₖ).

Essa normalização evita valores excessivamente altos, reduz a saturação da função softmax e garante uma distribuição adequada dos pesos de atenção.

## 📊 Exemplo de input e output esperado
Entrada
```
Q = [[1, 0],
     [0, 1]]

K = [[1, 0],
     [0, 1]]

V = [[1, 2],
     [3, 4]]
```

Output esperado (valores aproximados)
```
[[1.88 2.88]
 [2.12 3.12]]
```

Cada linha do output representa o resultado da atenção aplicada a uma query, combinando os valores da matriz V de acordo com os pesos calculados pelo mecanismo de Self-Attention.

## 📂 Estrutura do repositório
```
.
├── attention.py
├── test_attention.py
├── requirements.txt
└── README.md
```
