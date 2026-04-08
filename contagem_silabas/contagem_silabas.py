#!/usr/bin/env python3
"""
Script para contagem de sílabas em textos usando NLTK.
Processa múltiplos datasets e gera estatísticas em CSV.
"""
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
import numpy as np


def download_nltk_resources():
    """Baixa recursos necessários do NLTK."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Baixando recursos do NLTK...")
        nltk.download('punkt')


def contar_silabas_nltk(palavra):
    """
    Conta o número de sílabas em uma palavra em português usando heurísticas.
    Adaptado para português.

    Args:
        palavra (str): Palavra para contar sílabas

    Returns:
        int: Número de sílabas
    """
    palavra = palavra.lower().strip()
    if not palavra:
        return 0

    # Remove caracteres não alfabéticos
    palavra = re.sub(r'[^a-záàâãéèêíîóôõúç]', '', palavra)

    if not palavra:
        return 0

    # Padrões de vogais em português (ç é consoante, não vogal)
    vogais = 'aeiouáàâãéèêíîóôõú'

    # Conta grupos de vogais como uma sílaba
    silabas = 0
    i = 0
    while i < len(palavra):
        if palavra[i] in vogais:
            silabas += 1
            # Pula vogais consecutivas (ditongos, tritongos)
            while i + 1 < len(palavra) and palavra[i + 1] in vogais:
                # Casos especiais de hiatos em português
                if (palavra[i:i+2] in ['ia', 'ie', 'io', 'ua', 'ue', 'ui', 'uo'] and
                    i > 0 and palavra[i-1] not in vogais):
                    # Pode ser hiato, conta como sílaba separada
                    break
                i += 1
        i += 1

    # Ajustes para casos especiais em português
    if silabas == 0:
        silabas = 1

    # Ajuste para palavras terminadas em 'e' átono
    if palavra.endswith('e') and silabas > 1 and palavra[-2] not in vogais:
        # Verifica se é um 'e' átono final
        if not any(acento in palavra[-3:] for acento in 'áàâãéèêíîóôõúç'):
            pass  # Mantém a contagem (em português o 'e' final geralmente é pronunciado)

    return silabas


def processar_arquivo_texto(caminho_arquivo):
    """
    Processa um arquivo de texto e retorna estatísticas de sílabas.

    Args:
        caminho_arquivo (str): Caminho para o arquivo de texto

    Returns:
        dict: Dicionário com estatísticas ou None se erro
    """
    if not os.path.exists(caminho_arquivo):
        return None

    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            texto = f.read().strip()
    except Exception as e:
        print(f"Erro ao ler {caminho_arquivo}: {e}")
        return None

    if not texto:
        return None

    # Tokeniza em sentenças usando NLTK
    try:
        sentencas = sent_tokenize(texto, language='portuguese')
    except:
        # Fallback se não tiver o modelo português
        sentencas = sent_tokenize(texto)

    if not sentencas:
        return None

    # Calcula estatísticas
    silabas_por_sentenca = []
    total_silabas = 0
    total_palavras = 0

    for sentenca in sentencas:
        # Tokeniza palavras da sentença
        palavras = word_tokenize(sentenca.lower())
        # Remove pontuação
        palavras = [palavra for palavra in palavras if palavra.isalpha()]

        silabas_sentenca = 0
        for palavra in palavras:
            silabas = contar_silabas_nltk(palavra)
            silabas_sentenca += silabas
            total_silabas += silabas
            total_palavras += 1

        if silabas_sentenca > 0:
            silabas_por_sentenca.append(silabas_sentenca)

    if not silabas_por_sentenca:
        return None

    # Calcula médias
    media_silabas_por_sentenca = sum(silabas_por_sentenca) / len(silabas_por_sentenca)
    media_silabas_por_palavra = total_silabas / total_palavras if total_palavras > 0 else 0

    return {
        'total_sentencas': len(sentencas),
        'total_palavras': total_palavras,
        'total_silabas': total_silabas,
        'media_silabas_por_sentenca': media_silabas_por_sentenca,
        'media_silabas_por_palavra': media_silabas_por_palavra,
        'silabas_por_sentenca': silabas_por_sentenca
    }


def processar_dataset(nome_dataset, caminho_dataset, verbose=True):
    """
    Processa um dataset específico.

    Args:
        nome_dataset (str): Nome do dataset
        caminho_dataset (str): Caminho para o dataset
        verbose (bool): Se deve imprimir progresso

    Returns:
        list: Lista de resultados processados
    """
    resultados = []

    if verbose:
        print(f"\nProcessando dataset: {nome_dataset}")
        print(f"Caminho: {caminho_dataset}")

    if not os.path.exists(caminho_dataset):
        print(f"AVISO: Diretório não encontrado: {caminho_dataset}")
        return resultados

    # Conta arquivos no dataset
    arquivos_processados = 0

    # Processa todos os arquivos .txt no diretório
    for root, dirs, files in os.walk(caminho_dataset):
        for file in files:
            if file.endswith('.txt'):
                caminho_arquivo = os.path.join(root, file)

                # Processa o arquivo
                stats = processar_arquivo_texto(caminho_arquivo)

                if stats:
                    # Extrai informações do caminho relativo
                    caminho_relativo = os.path.relpath(caminho_arquivo, caminho_dataset)
                    partes_caminho = caminho_relativo.split(os.sep)

                    # Determina subset se existir
                    subset = ""
                    if len(partes_caminho) > 1:
                        subset = partes_caminho[0]

                    resultado = {
                        'arquivo': file,
                        'dataset': nome_dataset,
                        'subset': subset,
                        'caminho_relativo': caminho_relativo,
                        'caminho_completo': caminho_arquivo,
                        'total_sentencas': stats['total_sentencas'],
                        'total_palavras': stats['total_palavras'],
                        'total_silabas': stats['total_silabas'],
                        'media_silabas_por_sentenca': stats['media_silabas_por_sentenca'],
                        'media_silabas_por_palavra': stats['media_silabas_por_palavra']
                    }
                    resultados.append(resultado)
                    arquivos_processados += 1

                    # Progresso a cada 50 arquivos
                    if verbose and arquivos_processados % 50 == 0:
                        print(f"  Processados {arquivos_processados} arquivos...")

    if verbose:
        print(f"  Total de arquivos processados em {nome_dataset}: {arquivos_processados}")

    return resultados


def processar_todos_datasets(datasets, verbose=True):
    """
    Processa todos os datasets e coleta estatísticas de sílabas.

    Args:
        datasets (dict): Dicionário com nome e caminho dos datasets
        verbose (bool): Se deve imprimir progresso

    Returns:
        list: Lista com todos os resultados
    """
    resultados = []

    for nome_dataset, caminho_dataset in datasets.items():
        resultados_dataset = processar_dataset(nome_dataset, caminho_dataset, verbose)
        resultados.extend(resultados_dataset)

    if verbose:
        print(f"\nTotal geral de arquivos processados: {len(resultados)}")

    return resultados


def gerar_csv_agrupado(df_resultados, nome_arquivo_base):
    """
    Gera um CSV agrupado por tipo (Human vs LLM).

    Args:
        df_resultados (pd.DataFrame): DataFrame com os resultados
        nome_arquivo_base (str): Nome base do arquivo CSV
    """
    # Cria colunas derivadas para análise
    df_resultados['tipo'] = df_resultados['dataset'].apply(
        lambda x: 'Human' if 'Human' in x else 'LLM'
    )

    # Agrupa por tipo e calcula estatísticas
    agrupado = df_resultados.groupby('tipo').agg({
        'total_sentencas': 'sum',
        'total_palavras': 'sum',
        'total_silabas': 'sum',
        'media_silabas_por_sentenca': 'mean',
        'media_silabas_por_palavra': 'mean'
    }).round(3)

    # Calcula médias ponderadas mais precisas
    for tipo in ['Human', 'LLM']:
        df_tipo = df_resultados[df_resultados['tipo'] == tipo]
        total_palavras = agrupado.loc[tipo, 'total_palavras']
        total_silabas = agrupado.loc[tipo, 'total_silabas']

        if total_palavras > 0:
            agrupado.loc[tipo, 'media_silabas_por_palavra'] = total_silabas / total_palavras

    # Salva CSV agrupado
    nome_agrupado = nome_arquivo_base.replace('.csv', '_agrupado.csv')
    agrupado.to_csv(nome_agrupado, encoding='utf-8')

    print(f"CSV agrupado salvo em {nome_agrupado}")
    print("\nEstatísticas agrupadas:")
    print(agrupado)

    return agrupado


def realizar_testes_estatisticos(df_resultados):
    """
    Realiza testes estatísticos de significância para comparar Human vs LLM.

    Args:
        df_resultados (pd.DataFrame): DataFrame com os resultados

    Returns:
        dict: Dicionário com resultados dos testes
    """
    print("\n" + "=" * 60)
    print("ANÁLISE ESTATÍSTICA DE SIGNIFICÂNCIA")
    print("=" * 60)

    # Separa dados por tipo
    df_human = df_resultados[df_resultados['tipo'] == 'Human']
    df_llm = df_resultados[df_resultados['tipo'] == 'LLM']

    metricas = ['media_silabas_por_sentenca', 'media_silabas_por_palavra']
    resultados_testes = {}

    for metrica in metricas:
        dados_human = df_human[metrica].dropna()
        dados_llm = df_llm[metrica].dropna()

        print(f"\n--- {metrica} ---")
        print(f"Human: n={len(dados_human)}, média={dados_human.mean():.4f}, std={dados_human.std():.4f}")
        print(f"LLM:   n={len(dados_llm)}, média={dados_llm.mean():.4f}, std={dados_llm.std():.4f}")

        # Teste de normalidade (Shapiro-Wilk) - usa amostra se dados muito grandes
        amostra_human = dados_human.sample(min(5000, len(dados_human)), random_state=42)
        amostra_llm = dados_llm.sample(min(5000, len(dados_llm)), random_state=42)

        _, p_shapiro_human = stats.shapiro(amostra_human)
        _, p_shapiro_llm = stats.shapiro(amostra_llm)

        print("\nTeste de Normalidade (Shapiro-Wilk):")
        print(f"  Human: p-value = {p_shapiro_human:.6f} {'(normal)' if p_shapiro_human > 0.05 else '(não-normal)'}")
        print(f"  LLM:   p-value = {p_shapiro_llm:.6f} {'(normal)' if p_shapiro_llm > 0.05 else '(não-normal)'}")

        # Teste t independente (paramétrico)
        t_stat, p_ttest = stats.ttest_ind(dados_human, dados_llm)
        print("\nTeste t independente:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value = {p_ttest:.6e}")
        print(f"  Significativo (α=0.05): {'SIM' if p_ttest < 0.05 else 'NÃO'}")
        print(f"  Significativo (α=0.01): {'SIM' if p_ttest < 0.01 else 'NÃO'}")

        # Teste de Mann-Whitney U (não-paramétrico)
        u_stat, p_mannwhitney = stats.mannwhitneyu(dados_human, dados_llm, alternative='two-sided')
        print("\nTeste Mann-Whitney U (não-paramétrico):")
        print(f"  U-statistic = {u_stat:.4f}")
        print(f"  p-value = {p_mannwhitney:.6e}")
        print(f"  Significativo (α=0.05): {'SIM' if p_mannwhitney < 0.05 else 'NÃO'}")
        print(f"  Significativo (α=0.01): {'SIM' if p_mannwhitney < 0.01 else 'NÃO'}")

        # Calcula tamanho do efeito (Cohen's d)
        pooled_std = np.sqrt(((len(dados_human) - 1) * dados_human.std()**2 +
                              (len(dados_llm) - 1) * dados_llm.std()**2) /
                             (len(dados_human) + len(dados_llm) - 2))
        cohens_d = (dados_human.mean() - dados_llm.mean()) / pooled_std

        # Interpretação do tamanho do efeito
        if abs(cohens_d) < 0.2:
            interpretacao = "negligível"
        elif abs(cohens_d) < 0.5:
            interpretacao = "pequeno"
        elif abs(cohens_d) < 0.8:
            interpretacao = "médio"
        else:
            interpretacao = "grande"

        print("\nTamanho do Efeito (Cohen's d):")
        print(f"  d = {cohens_d:.4f} ({interpretacao})")

        resultados_testes[metrica] = {
            'human_mean': dados_human.mean(),
            'human_std': dados_human.std(),
            'human_n': len(dados_human),
            'llm_mean': dados_llm.mean(),
            'llm_std': dados_llm.std(),
            'llm_n': len(dados_llm),
            't_statistic': t_stat,
            'p_ttest': p_ttest,
            'u_statistic': u_stat,
            'p_mannwhitney': p_mannwhitney,
            'cohens_d': cohens_d,
            'effect_size': interpretacao
        }

    # Análise por corpus (FakeBR e FakeTrue separadamente)
    print("\n" + "=" * 60)
    print("ANÁLISE POR CORPUS")
    print("=" * 60)

    for corpus in ['FakeBR', 'FakeTrue']:
        df_corpus = df_resultados[df_resultados['corpus'] == corpus]
        df_corpus_human = df_corpus[df_corpus['tipo'] == 'Human']
        df_corpus_llm = df_corpus[df_corpus['tipo'] == 'LLM']

        print(f"\n>>> Corpus: {corpus}")

        for metrica in metricas:
            dados_human = df_corpus_human[metrica].dropna()
            dados_llm = df_corpus_llm[metrica].dropna()

            if len(dados_human) > 0 and len(dados_llm) > 0:
                t_stat, p_ttest = stats.ttest_ind(dados_human, dados_llm)
                u_stat, p_mannwhitney = stats.mannwhitneyu(dados_human, dados_llm, alternative='two-sided')

                # Cohen's d para o corpus
                pooled_std = np.sqrt((dados_human.std()**2 + dados_llm.std()**2) / 2)
                cohens_d = (dados_human.mean() - dados_llm.mean()) / pooled_std if pooled_std > 0 else 0

                print(f"\n  {metrica}:")
                print(f"    Human: média={dados_human.mean():.4f}, std={dados_human.std():.4f}, n={len(dados_human)}")
                print(f"    LLM:   média={dados_llm.mean():.4f}, std={dados_llm.std():.4f}, n={len(dados_llm)}")
                print(f"    Teste t: p={p_ttest:.6e} {'***' if p_ttest < 0.001 else '**' if p_ttest < 0.01 else '*' if p_ttest < 0.05 else ''}")
                print(f"    Mann-Whitney: p={p_mannwhitney:.6e} {'***' if p_mannwhitney < 0.001 else '**' if p_mannwhitney < 0.01 else '*' if p_mannwhitney < 0.05 else ''}")
                print(f"    Cohen's d: {cohens_d:+.4f}")

    print("\n" + "=" * 60)
    print("Legenda: *** p<0.001, ** p<0.01, * p<0.05")
    print("=" * 60)

    return resultados_testes


def salvar_resultados(resultados, nome_arquivo="estatisticas_silabas_nltk.csv", verbose=True):
    """
    Salva os resultados em CSV e gera estatísticas.

    Args:
        resultados (list): Lista de resultados
        nome_arquivo (str): Nome do arquivo CSV
        verbose (bool): Se deve imprimir estatísticas

    Returns:
        pd.DataFrame: DataFrame com os resultados
    """
    if not resultados:
        print("Nenhum resultado para salvar.")
        return None

    # Cria DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Salva o DataFrame como CSV
    df_resultados.to_csv(nome_arquivo, index=False, encoding='utf-8')

    if verbose:
        print(f"\nResultados salvos em {nome_arquivo}")
        print(f"Total de linhas: {len(df_resultados)}")

        # Cria colunas derivadas para análise
        df_resultados['tipo'] = df_resultados['dataset'].apply(
            lambda x: 'Human' if 'Human' in x else 'LLM'
        )
        df_resultados['corpus'] = df_resultados['dataset'].apply(
            lambda x: 'FakeBR' if 'FakeBR' in x else 'FakeTrue'
        )

        print("\nEstatísticas por dataset:")
        stats_dataset = df_resultados.groupby('dataset')[
            ['media_silabas_por_sentenca', 'media_silabas_por_palavra']
        ].describe()
        print(stats_dataset)

        # Resumo por dataset
        resumo_dataset = df_resultados.groupby('dataset').agg({
            'media_silabas_por_sentenca': ['mean', 'std', 'count'],
            'media_silabas_por_palavra': ['mean', 'std', 'count'],
            'total_sentencas': 'sum',
            'total_palavras': 'sum',
            'total_silabas': 'sum'
        }).round(3)

        # Salva resumo
        nome_resumo = nome_arquivo.replace('.csv', '_resumo.csv')
        resumo_dataset.to_csv(nome_resumo, encoding='utf-8')
        print(f"\nResumo salvo em {nome_resumo}")

        # Gera CSV agrupado
        gerar_csv_agrupado(df_resultados, nome_arquivo)

        # Comparações por tipo
        print("\n=== COMPARAÇÕES ===")
        print("\nMédia de sílabas por sentença por dataset:")
        medias_sentenca = df_resultados.groupby('dataset')['media_silabas_por_sentenca'].mean().sort_values(ascending=False)
        print(medias_sentenca)

        print("\nMédia de sílabas por palavra por dataset:")
        medias_palavra = df_resultados.groupby('dataset')['media_silabas_por_palavra'].mean().sort_values(ascending=False)
        print(medias_palavra)

        # Análise por tipo (Human vs LLM)
        print("\n=== ANÁLISE HUMAN vs LLM ===")
        comparacao_tipo = df_resultados.groupby('tipo')[
            ['media_silabas_por_sentenca', 'media_silabas_por_palavra']
        ].mean()
        print("\nMédias gerais por tipo:")
        print(comparacao_tipo)

        # Realiza testes estatísticos de significância
        resultados_testes = realizar_testes_estatisticos(df_resultados)

        # Salva resultados dos testes estatísticos em CSV
        nome_testes = nome_arquivo.replace('.csv', '_testes_estatisticos.csv')
        df_testes = pd.DataFrame(resultados_testes).T
        df_testes.to_csv(nome_testes, encoding='utf-8')
        print(f"\nResultados dos testes estatísticos salvos em {nome_testes}")

    return df_resultados


def testar_contagem_silabas():
    """Testa a função de contagem de sílabas com palavras exemplo."""
    print("=== TESTE DA FUNÇÃO DE CONTAGEM DE SÍLABAS ===")
    palavras_teste = [
        'casa', 'computador', 'análise', 'inteligência',
        'artificial', 'Brasil', 'tecnologia', 'universidade'
    ]

    for palavra in palavras_teste:
        silabas = contar_silabas_nltk(palavra)
        print(f"{palavra}: {silabas} sílabas")

    print("=" * 50)


def main():
    """Função principal do script."""

    # Baixa recursos do NLTK
    download_nltk_resources()

    # Caminhos relativos ao próprio script (funciona de qualquer cwd)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CORPUS_DIR = os.path.join(os.path.dirname(BASE_DIR), "corpus")

    datasets = {
        'FakeBR_Human': os.path.join(CORPUS_DIR, "Fake.br-Corpus-master", "full_texts", "fake_br_clean"),
        'FakeBR_LLM': os.path.join(CORPUS_DIR, "fake-news-llm-ptbr-main", "fake-news-llm-ptbr-main", "data", "Fake.Br"),
        'FakeTrue_Human': os.path.join(CORPUS_DIR, "FakeTrue.Br-main", "fake"),
        'FakeTrue_LLM': os.path.join(CORPUS_DIR, "fake-news-llm-ptbr-main", "fake-news-llm-ptbr-main", "data", "FakeTrueBR"),
    }

    verbose = True
    arquivo_saida = os.path.join(BASE_DIR, "resultados", "estatisticas_silabas_nltk.csv")
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    print("Iniciando processamento de todos os datasets...")
    testar_contagem_silabas()

    # Processa todos os datasets
    resultados = processar_todos_datasets(datasets, verbose)

    # Salva resultados
    df_resultados = salvar_resultados(resultados, arquivo_saida, verbose)

    if df_resultados is not None:
        print(f"\nProcessamento concluído! Dados salvos em {arquivo_saida}")


if __name__ == "__main__":
    main()