"""
Script para análise agregada de POS tags: Human Total vs LLM Total.
Combina FakeBR + FakeTrue para comparar todos os textos humanos contra todos os textos de LLM.
"""

import os
from collections import Counter
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def ler_tags_de_arquivo(filepath):
    """Lê tags POS de um arquivo de resultado do tagger."""
    tags = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith('#'):
                continue
            partes = linha.split('\t')
            if len(partes) >= 3:
                tags.append(partes[2])  # terceira coluna = POS-tag
    return tags


def analisar_diretorio(diretorio):
    """Analisa um diretório e conta as tags POS."""
    contador = Counter()
    total_tokens = 0
    arquivos = 0

    for root, _, files in os.walk(diretorio):
        for nome in files:
            if nome.endswith('.txt'):
                caminho = os.path.join(root, nome)
                tags = ler_tags_de_arquivo(caminho)
                contador.update(tags)
                total_tokens += len(tags)
                arquivos += 1

    media = total_tokens / arquivos if arquivos else 0

    return {
        "diretorio": diretorio,
        "total_tokens": total_tokens,
        "total_arquivos": arquivos,
        "media_por_arquivo": media,
        "frequencia_tags": contador
    }


def combinar_resultados(nome, resultados):
    """Combina múltiplos resultados em um único."""
    contador = Counter()
    total_tokens = 0
    total_arquivos = 0

    for r in resultados:
        contador.update(r["frequencia_tags"])
        total_tokens += r["total_tokens"]
        total_arquivos += r["total_arquivos"]

    media = total_tokens / total_arquivos if total_arquivos else 0

    return {
        "diretorio": nome,
        "total_tokens": total_tokens,
        "total_arquivos": total_arquivos,
        "media_por_arquivo": media,
        "frequencia_tags": contador
    }


def imprimir_analise(resultado, top_k=10):
    """Imprime análise de um resultado."""
    print(f"Análise: {resultado['diretorio']}")
    print(f"  - Arquivos: {resultado['total_arquivos']}")
    print(f"  - Tokens: {resultado['total_tokens']}")
    print(f"  - Média de tokens/arquivo: {resultado['media_por_arquivo']:.2f}")
    print(f"  - Top {top_k} tags:")
    for tag, count in resultado["frequencia_tags"].most_common(top_k):
        print(f"    {tag}: {count}")


def salvar_resultados_csv(lista_resultados, caminho_csv):
    """Salva resultados em CSV."""
    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    with open(caminho_csv, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["conjunto", "total_tokens", "total_arquivos", "media_por_arquivo", "tag", "frequencia"])
        for resultado in lista_resultados:
            conjunto = resultado["diretorio"]
            total_tokens = resultado["total_tokens"]
            total_arquivos = resultado["total_arquivos"]
            media_por_arquivo = resultado["media_por_arquivo"]
            for tag, freq in resultado["frequencia_tags"].items():
                writer.writerow([conjunto, total_tokens, total_arquivos, media_por_arquivo, tag, freq])


def salvar_tabela_tags_por_conjunto(resultado, pasta_saida="tabelas_detalhadas"):
    """Salva tabela detalhada de tags por conjunto."""
    os.makedirs(pasta_saida, exist_ok=True)

    conjunto = resultado["diretorio"]
    total_arquivos = resultado["total_arquivos"]
    frequencias = resultado["frequencia_tags"]

    dados = []
    for tag, total in frequencias.items():
        media_por_arquivo = total / total_arquivos if total_arquivos > 0 else 0
        dados.append({
            "tag": tag,
            "total_ocorrencias": total,
            "media_por_arquivo": media_por_arquivo
        })

    df = pd.DataFrame(dados)
    if not df.empty:
        df = df.sort_values(by="total_ocorrencias", ascending=False)

    nome_arquivo = conjunto.lower().replace(" ", "_").replace(".", "").replace("-", "_") + "_tags_detalhado.csv"
    caminho = os.path.join(pasta_saida, nome_arquivo)

    df.to_csv(caminho, index=False, encoding="utf-8-sig")
    print(f"Tabela detalhada salva: {caminho}")
    return caminho


# =============================================================================
# FUNÇÕES DE GRÁFICOS
# =============================================================================

def grafico_barras_agrupadas(csv_paths, nomes, tipo="frequencia_relativa", salvar_em=None, ordem_tags=None, palette=None):
    """Gera gráfico de barras agrupadas comparando conjuntos."""
    dfs = []
    for path, nome in zip(csv_paths, nomes):
        df = pd.read_csv(path)
        total = df["total_ocorrencias"].sum()
        if tipo == "frequencia_relativa":
            df[tipo] = df["total_ocorrencias"] / total
        df["conjunto"] = nome
        dfs.append(df[["tag", tipo, "conjunto"]])

    df_final = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df_final,
        x="tag",
        y=tipo,
        hue="conjunto",
        order=ordem_tags,
        palette=palette
    )

    plt.ylabel("Frequência Relativa" if tipo == "frequencia_relativa" else "Total", fontsize=14)
    plt.xlabel("Classe Gramatical (POS)", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Conjunto", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)
        arqs_names = "_".join([name.replace(" ", "_") for name in nomes])
        nome_arquivo = f"barras_comparativas_{tipo}_{arqs_names}.png"
        plt.savefig(os.path.join(salvar_em, nome_arquivo), dpi=150)
        print(f"Gráfico salvo: {os.path.join(salvar_em, nome_arquivo)}")
    plt.close()


def grafico_diferencas(df_diferencas, salvar_em=None):
    """Gera gráfico de barras horizontais mostrando diferenças."""
    plt.figure(figsize=(12, 10))

    df_plot = df_diferencas.sort_values("diferenca")
    colors = ['#1f77b4' if x < 0 else '#ff7f0e' for x in df_plot["diferenca"]]  # azul/laranja

    plt.barh(df_plot["tag"], df_plot["diferenca"], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel("Diferença de Frequência Relativa", fontsize=14)
    plt.ylabel("Classe Gramatical (POS)", fontsize=14)
    plt.title("Diferença de Frequência Relativa: Humano vs Máquina\n(Azul = mais frequente em Humano, Laranja = mais frequente em Máquina)", fontsize=12)
    plt.tight_layout()

    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)
        plt.savefig(os.path.join(salvar_em, "diferencas_frequencia_relativa.png"), dpi=150)
        print(f"Gráfico de diferenças salvo: {os.path.join(salvar_em, 'diferencas_frequencia_relativa.png')}")
    plt.close()


# =============================================================================
# FUNÇÕES DE ANÁLISE ESTATÍSTICA
# =============================================================================

def calcular_diferencas_frequencia_relativa(csv1, csv2, nome1, nome2):
    """Calcula as diferenças de frequência relativa entre dois conjuntos."""
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    total1 = df1["total_ocorrencias"].sum()
    total2 = df2["total_ocorrencias"].sum()

    df1["freq_rel"] = df1["total_ocorrencias"] / total1
    df2["freq_rel"] = df2["total_ocorrencias"] / total2

    df1 = df1.set_index("tag")
    df2 = df2.set_index("tag")

    todas_tags = sorted(set(df1.index) | set(df2.index))

    resultados = []
    for tag in todas_tags:
        freq1 = df1.loc[tag, "freq_rel"] if tag in df1.index else 0
        freq2 = df2.loc[tag, "freq_rel"] if tag in df2.index else 0
        diff = freq2 - freq1  # positivo = mais frequente em LLM
        resultados.append({
            "tag": tag,
            f"freq_{nome1}": freq1,
            f"freq_{nome2}": freq2,
            "diferenca": diff,
            "diferenca_abs": abs(diff)
        })

    df_result = pd.DataFrame(resultados).sort_values("diferenca_abs", ascending=False)
    return df_result


def realizar_testes_estatisticos(csv1, csv2, nome1, nome2, salvar_em=None):
    """
    Realiza testes estatísticos para comparar distribuições de POS tags.

    Inclui:
    - Teste qui-quadrado (significância estatística)
    - V de Cramér (tamanho do efeito - MAIS IMPORTANTE com amostras grandes)

    NOTA: Com amostras grandes, o qui-quadrado quase sempre será significativo.
    O V de Cramér é a métrica que realmente indica se a diferença é relevante na prática.
    """
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    df1 = df1.set_index("tag")
    df2 = df2.set_index("tag")

    todas_tags = sorted(set(df1.index) | set(df2.index))

    obs1 = [df1.loc[tag, "total_ocorrencias"] if tag in df1.index else 0 for tag in todas_tags]
    obs2 = [df2.loc[tag, "total_ocorrencias"] if tag in df2.index else 0 for tag in todas_tags]

    # Tabela de contingência
    tabela = np.array([obs1, obs2])

    # Teste qui-quadrado
    chi2, p_value, dof, expected = stats.chi2_contingency(tabela)

    # V de Cramér (tamanho do efeito)
    n = tabela.sum()
    min_dim = min(tabela.shape[0] - 1, tabela.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    # Interpretação do V de Cramér
    if cramers_v < 0.1:
        interpretacao_v = "negligivel"
    elif cramers_v < 0.3:
        interpretacao_v = "pequeno"
    elif cramers_v < 0.5:
        interpretacao_v = "medio"
    else:
        interpretacao_v = "grande"

    # Impressão dos resultados
    print("\n" + "=" * 60)
    print("TESTES ESTATISTICOS - DISTRIBUICAO DE POS TAGS")
    print("=" * 60)
    print(f"Comparacao: {nome1} vs {nome2}")
    print(f"Total de tokens {nome1}: {sum(obs1):,}")
    print(f"Total de tokens {nome2}: {sum(obs2):,}")
    print(f"\n--- Teste Qui-Quadrado ---")
    print(f"Chi2 = {chi2:.4f}")
    print(f"p-value = {p_value:.2e}")
    print(f"Graus de liberdade = {dof}")
    print(f"Significativo (alpha=0.05): {'SIM' if p_value < 0.05 else 'NAO'}")
    print(f"\n--- V de Cramer (Tamanho do Efeito) ---")
    print(f"V = {cramers_v:.4f} ({interpretacao_v})")
    print(f"\nNOTA: Com amostras grandes, o qui-quadrado tende a ser sempre")
    print(f"significativo. O V de Cramer indica a relevancia pratica.")
    print("=" * 60)

    # Preparar resultados para salvar
    resultados = {
        'comparacao': f"{nome1} vs {nome2}",
        'n_tokens_grupo1': sum(obs1),
        'n_tokens_grupo2': sum(obs2),
        'n_categorias': len(todas_tags),
        'chi2': chi2,
        'p_value': p_value,
        'graus_liberdade': dof,
        'significativo_005': 'SIM' if p_value < 0.05 else 'NAO',
        'significativo_001': 'SIM' if p_value < 0.01 else 'NAO',
        'cramers_v': cramers_v,
        'interpretacao_efeito': interpretacao_v
    }

    # Salvar em CSV
    if salvar_em:
        os.makedirs(salvar_em, exist_ok=True)
        caminho_csv = os.path.join(salvar_em, "testes_estatisticos.csv")
        df_resultados = pd.DataFrame([resultados])
        df_resultados.to_csv(caminho_csv, index=False, encoding='utf-8-sig')
        print(f"\nResultados salvos em: {caminho_csv}")

    return resultados


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do script."""

    # Diretório base = pasta deste script (funciona de qualquer cwd)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TAGGER_RESULTS = os.path.join(BASE_DIR, "tagger_results")

    # Diretórios Human
    fake_br_human = os.path.join(TAGGER_RESULTS, "FakeHuman", "FakeBR")
    fake_true_human = os.path.join(TAGGER_RESULTS, "FakeHuman", "FakeTrueBR")

    # Diretórios LLM
    fake_br_llm_train = os.path.join(TAGGER_RESULTS, "FakeLLM", "Fake.Br", "train")
    fake_br_llm_test = os.path.join(TAGGER_RESULTS, "FakeLLM", "Fake.Br", "test")
    fake_true_llm_train = os.path.join(TAGGER_RESULTS, "FakeLLM", "FakeTrueBR", "train")
    fake_true_llm_test = os.path.join(TAGGER_RESULTS, "FakeLLM", "FakeTrueBR", "test")

    # Diretório de saída
    OUTPUT_DIR = os.path.join(BASE_DIR, "resultados_agregados")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ANÁLISE AGREGADA: HUMAN TOTAL vs LLM TOTAL")
    print("=" * 60)

    # ==========================================================================
    # ANÁLISE DOS DADOS HUMAN
    # ==========================================================================
    print("\n>>> Analisando dados HUMAN...")

    resultado_fakebr_human = analisar_diretorio(fake_br_human)
    resultado_fakebr_human["diretorio"] = "FakeBR_Human"

    resultado_faketrue_human = analisar_diretorio(fake_true_human)
    resultado_faketrue_human["diretorio"] = "FakeTrue_Human"

    resultado_human_total = combinar_resultados("Human_Total", [
        resultado_fakebr_human,
        resultado_faketrue_human
    ])

    print("\n=== HUMAN TOTAL ===")
    imprimir_analise(resultado_human_total)

    # ==========================================================================
    # ANÁLISE DOS DADOS LLM
    # ==========================================================================
    print("\n>>> Analisando dados LLM...")

    resultado_fakebr_llm_train = analisar_diretorio(fake_br_llm_train)
    resultado_fakebr_llm_test = analisar_diretorio(fake_br_llm_test)
    resultado_fakebr_llm = combinar_resultados("FakeBR_LLM", [
        resultado_fakebr_llm_train,
        resultado_fakebr_llm_test
    ])

    resultado_faketrue_llm_train = analisar_diretorio(fake_true_llm_train)
    resultado_faketrue_llm_test = analisar_diretorio(fake_true_llm_test)
    resultado_faketrue_llm = combinar_resultados("FakeTrue_LLM", [
        resultado_faketrue_llm_train,
        resultado_faketrue_llm_test
    ])

    resultado_llm_total = combinar_resultados("LLM_Total", [
        resultado_fakebr_llm,
        resultado_faketrue_llm
    ])

    print("\n=== LLM TOTAL ===")
    imprimir_analise(resultado_llm_total)

    # ==========================================================================
    # SALVAR TABELAS DETALHADAS
    # ==========================================================================
    print("\n>>> Salvando tabelas detalhadas...")

    tabelas_dir = os.path.join(OUTPUT_DIR, "tabelas")
    csv_human = salvar_tabela_tags_por_conjunto(resultado_human_total, pasta_saida=tabelas_dir)
    csv_llm = salvar_tabela_tags_por_conjunto(resultado_llm_total, pasta_saida=tabelas_dir)

    # Salvar CSV geral
    salvar_resultados_csv(
        [resultado_human_total, resultado_llm_total],
        os.path.join(OUTPUT_DIR, "resultados_agregados.csv")
    )
    print(f"Resultados agregados salvos em: {os.path.join(OUTPUT_DIR, 'resultados_agregados.csv')}")

    # ==========================================================================
    # GERAR GRÁFICOS
    # ==========================================================================
    print("\n>>> Gerando gráficos...")

    graficos_dir = os.path.join(OUTPUT_DIR, "graficos")
    csvs = [csv_human, csv_llm]
    nomes = ["Humano", "Máquina"]

    palette = {
        "Humano": "#1f77b4",    # azul
        "Máquina": "#ff7f0e"    # laranja
    }

    ordem_tags = ["NOUN", "ADP", "PUNCT", "VERB", "DET", "PROPN", "ADJ", "ADV",
                  "AUX", "PRON", "CCONJ", "NUM", "SCONJ", "SYM", "INTJ", "X"]

    # Gráfico de barras - Frequência Relativa
    grafico_barras_agrupadas(
        csvs, nomes,
        tipo="frequencia_relativa",
        salvar_em=graficos_dir,
        ordem_tags=ordem_tags,
        palette=palette
    )

    # Gráfico de barras - Total de Ocorrências
    grafico_barras_agrupadas(
        csvs, nomes,
        tipo="total_ocorrencias",
        salvar_em=graficos_dir,
        ordem_tags=ordem_tags,
        palette=palette
    )

    # ==========================================================================
    # ANÁLISE DE DIFERENÇAS
    # ==========================================================================
    print("\n>>> Calculando diferenças...")

    df_diferencas = calcular_diferencas_frequencia_relativa(csv_human, csv_llm, "Humano", "Máquina")

    print("\nTop 10 diferenças de frequência relativa (Human vs LLM):")
    print("(Diferença positiva = mais frequente em LLM, negativa = mais frequente em Human)\n")
    print(df_diferencas.head(10).to_string(index=False))

    # Salvar tabela de diferenças
    df_diferencas.to_csv(os.path.join(tabelas_dir, "diferencas_human_vs_llm.csv"), index=False)
    print(f"\nTabela de diferenças salva em: {os.path.join(tabelas_dir, 'diferencas_human_vs_llm.csv')}")

    # Gráfico de diferenças
    grafico_diferencas(df_diferencas, salvar_em=graficos_dir)

    # ==========================================================================
    # TESTES ESTATÍSTICOS
    # ==========================================================================
    print("\n>>> Realizando testes estatísticos...")

    realizar_testes_estatisticos(csv_human, csv_llm, "Human", "LLM", salvar_em=tabelas_dir)

    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)
    print(f"\nResultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
