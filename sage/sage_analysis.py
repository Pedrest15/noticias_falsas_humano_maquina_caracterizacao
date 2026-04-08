"""
SAGE (Sparse Additive Generative Model) para análise de termos distintivos.

Compara textos de duas classes (Humano vs IA) e identifica quais termos
são mais característicos de cada grupo.

Referência: Eisenstein et al. (2011) - "Sparse Additive Generative Models of Text"
"""

import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.special import logsumexp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalizar_texto(texto):
    """
    Normaliza o texto: lowercase, remove acentos e caracteres especiais.

    Args:
        texto (str): Texto original

    Returns:
        str: Texto normalizado
    """
    if not isinstance(texto, str):
        return ""

    # Converter para lowercase
    texto = texto.lower()

    # Remover acentos usando normalização Unicode
    # NFD decompõe caracteres acentuados em base + acento
    # Depois removemos os caracteres de categoria 'Mn' (Mark, Nonspacing)
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(char for char in texto if unicodedata.category(char) != 'Mn')

    # Substituir ç por c (caso não tenha sido tratado pela normalização)
    texto = texto.replace('ç', 'c')

    return texto


class SAGEModel:
    """
    Implementação do modelo SAGE para análise de termos distintivos.

    O SAGE modela a distribuição de palavras como:
        log P(w|classe) = η_classe + m - Z

    Onde:
        - m: distribuição de fundo (log-frequências globais)
        - η: desvios esparsos por classe
        - Z: constante de normalização (log-sum-exp)
    """

    def __init__(self, regularization=0.1):
        """
        Args:
            regularization: peso da regularização L1 (maior = mais esparsidade)
        """
        self.regularization = regularization
        self.background = None  # m - distribuição de fundo
        self.components = None  # η - desvios por classe
        self.classes = None
        self.vocab_size = None

    def _initialize(self, X, y):
        """Inicializa parâmetros do modelo."""
        self.vocab_size = X.shape[1]
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Converter para array denso
        X_dense = X.toarray() if hasattr(X, 'toarray') else X

        # Distribuição de fundo: log P(w) global
        total_counts = np.sum(X_dense, axis=0) + 1e-10  # smoothing
        self.background = np.log(total_counts) - np.log(np.sum(total_counts))

        # Inicializar desvios η próximos de zero
        self.components = np.random.normal(0, 0.01, (n_classes, self.vocab_size))

    def _objective(self, params, X, y):
        """Função objetivo: -log_likelihood + L1_penalty"""
        n_classes = len(self.classes)
        self.components = params.reshape(n_classes, self.vocab_size)

        X_dense = X.toarray() if hasattr(X, 'toarray') else X

        total_ll = 0
        for i, class_label in enumerate(self.classes):
            mask = (y == class_label)
            if np.any(mask):
                X_class = X_dense[mask]

                # log P(w|classe) = η + m - log_sum_exp(η + m)
                log_unnorm = self.components[i] + self.background
                log_Z = logsumexp(log_unnorm)
                log_probs = log_unnorm - log_Z

                # Log-likelihood
                total_ll += np.sum(X_class * log_probs)

        # Regularização L1
        l1_penalty = self.regularization * np.sum(np.abs(self.components))

        return -(total_ll - l1_penalty)

    def _gradient(self, params, X, y):
        """Gradiente analítico da função objetivo."""
        n_classes = len(self.classes)
        self.components = params.reshape(n_classes, self.vocab_size)

        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        grad = np.zeros_like(self.components)

        for i, class_label in enumerate(self.classes):
            mask = (y == class_label)
            if np.any(mask):
                X_class = X_dense[mask]

                # Contagens observadas
                observed = np.sum(X_class, axis=0)

                # Contagens esperadas sob o modelo
                log_unnorm = self.components[i] + self.background
                probs = np.exp(log_unnorm - logsumexp(log_unnorm))
                total_words = np.sum(X_class)
                expected = total_words * probs

                # Gradiente = observado - esperado - regularização
                grad[i] = observed - expected
                grad[i] -= self.regularization * np.sign(self.components[i])

        return -grad.flatten()

    def fit(self, X, y):
        """Treina o modelo SAGE."""
        self._initialize(X, y)

        result = minimize(
            fun=self._objective,
            x0=self.components.flatten(),
            args=(X, y),
            jac=self._gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )

        self.components = result.x.reshape(len(self.classes), self.vocab_size)
        return self

    def get_distinctive_terms(self, feature_names, top_k=30):
        """
        Extrai termos distintivos comparando as duas classes.

        Calcula: distintividade = η_classe0 - η_classe1
        - Valores positivos: mais característicos da classe 0 (humano)
        - Valores negativos: mais característicos da classe 1 (IA)
        """
        if len(self.classes) != 2:
            raise ValueError("Este método requer exatamente 2 classes")

        # Diferença entre os desvios das duas classes
        # Positivo = mais humano, Negativo = mais IA
        diff = self.components[0] - self.components[1]

        results = []

        # Top termos mais humanos (maiores valores positivos)
        top_human_idx = np.argsort(diff)[-top_k:][::-1]
        for idx in top_human_idx:
            if diff[idx] > 0.01:  # Threshold para significância
                results.append({
                    'termo': feature_names[idx],
                    'classe': 'humano',
                    'distintividade': diff[idx],
                    'eta_humano': self.components[0, idx],
                    'eta_ia': self.components[1, idx]
                })

        # Top termos mais IA (maiores valores negativos)
        top_ia_idx = np.argsort(diff)[:top_k]
        for idx in top_ia_idx:
            if diff[idx] < -0.01:  # Threshold para significância
                results.append({
                    'termo': feature_names[idx],
                    'classe': 'ia',
                    'distintividade': diff[idx],
                    'eta_humano': self.components[0, idx],
                    'eta_ia': self.components[1, idx]
                })

        return pd.DataFrame(results)


def carregar_corpus(diretorio):
    """Carrega todos os arquivos .txt de um diretório."""
    textos = {}
    diretorio = Path(diretorio)

    for arquivo in diretorio.rglob('*.txt'):
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                textos[arquivo.name] = f.read()
        except Exception as e:
            print(f"  Erro ao ler {arquivo.name}: {e}")

    return textos


def analisar_sage(dir_humano, dir_ia, output_dir, nome_corpus="corpus"):
    """
    Executa análise SAGE completa comparando dois corpus.

    Args:
        dir_humano: diretório com textos humanos
        dir_ia: diretório com textos de IA
        output_dir: diretório para salvar resultados
        nome_corpus: nome do corpus para identificação
    """
    print("=" * 80)
    print(f"ANÁLISE SAGE: {nome_corpus}")
    print("=" * 80)

    # 1. Carregar corpus
    print("\n[1/6] Carregando corpus...")
    corpus_humano = carregar_corpus(dir_humano)
    corpus_ia = carregar_corpus(dir_ia)

    print(f"  Textos humanos: {len(corpus_humano)}")
    print(f"  Textos IA: {len(corpus_ia)}")

    # 2. Parear arquivos (se necessário)
    nomes_comuns = set(corpus_humano.keys()) & set(corpus_ia.keys())

    if len(nomes_comuns) > 0:
        print(f"  Arquivos em comum (pareados): {len(nomes_comuns)}")
        # Usar apenas arquivos pareados
        dados = []
        for nome in nomes_comuns:
            dados.append((corpus_humano[nome], "humano"))
            dados.append((corpus_ia[nome], "ia"))
    else:
        print("  Sem arquivos pareados - usando todos os textos")
        dados = []
        for texto in corpus_humano.values():
            dados.append((texto, "humano"))
        for texto in corpus_ia.values():
            dados.append((texto, "ia"))

    if len(dados) == 0:
        print("\nERRO: Nenhum texto encontrado!")
        return None, None

    df = pd.DataFrame(dados, columns=["texto", "origem"])
    print(f"  Total de textos para análise: {len(df)}")

    # 3. Vetorização
    print("\n[2/6] Vetorizando textos...")
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=2000
    )

    X = vectorizer.fit_transform(df["texto"])

    le = LabelEncoder()
    y = le.fit_transform(df["origem"])

    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulário: {len(feature_names)} termos")
    print(f"  Matriz esparsa: {X.shape}")

    # 4. Treinar SAGE
    print("\n[3/6] Treinando modelo SAGE...")
    modelo = SAGEModel(regularization=0.1)
    modelo.fit(X, y)
    print("  Modelo treinado com sucesso!")

    # 5. Extrair termos distintivos
    print("\n[4/6] Extraindo termos distintivos...")
    resultados = modelo.get_distinctive_terms(feature_names, top_k=50)

    # Ordenar por distintividade absoluta
    resultados['distintividade_abs'] = resultados['distintividade'].abs()
    resultados = resultados.sort_values('distintividade_abs', ascending=False)

    n_humano = len(resultados[resultados['classe'] == 'humano'])
    n_ia = len(resultados[resultados['classe'] == 'ia'])
    print(f"  Termos distintivos humanos: {n_humano}")
    print(f"  Termos distintivos IA: {n_ia}")

    # 6. Salvar resultados
    print("\n[5/6] Salvando resultados...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV com todos os termos
    csv_file = output_path / f"sage_termos_{nome_corpus}.csv"
    resultados.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  CSV salvo: {csv_file}")

    # Relatório em texto
    report_file = output_path / f"sage_relatorio_{nome_corpus}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RELATÓRIO SAGE - {nome_corpus.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURAÇÃO:\n")
        f.write(f"  Diretório Humano: {dir_humano}\n")
        f.write(f"  Diretório IA: {dir_ia}\n")
        f.write(f"  Total de textos: {len(df)}\n")
        f.write(f"  Vocabulário: {len(feature_names)} termos\n")
        f.write(f"  Regularização L1: {modelo.regularization}\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETAÇÃO DOS RESULTADOS:\n")
        f.write("=" * 80 + "\n")
        f.write("  - Distintividade POSITIVA: termo mais frequente em textos HUMANOS\n")
        f.write("  - Distintividade NEGATIVA: termo mais frequente em textos de IA\n")
        f.write("  - Quanto maior o valor absoluto, mais distintivo o termo\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 20 TERMOS MAIS DISTINTIVOS DE HUMANOS\n")
        f.write("=" * 80 + "\n\n")

        humanos = resultados[resultados['classe'] == 'humano'].head(20)
        for i, row in enumerate(humanos.itertuples(), 1):
            f.write(f"  {i:2d}. {row.termo:<30} +{row.distintividade:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 20 TERMOS MAIS DISTINTIVOS DE IA\n")
        f.write("=" * 80 + "\n\n")

        ia = resultados[resultados['classe'] == 'ia'].head(20)
        for i, row in enumerate(ia.itertuples(), 1):
            f.write(f"  {i:2d}. {row.termo:<30} {row.distintividade:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Relatório salvo: {report_file}")

    # 7. Imprimir resumo
    print("\n[6/6] Resumo dos resultados:")
    print("\n" + "=" * 80)
    print("TOP 10 TERMOS MAIS DISTINTIVOS DE HUMANOS")
    print("=" * 80)
    for i, row in enumerate(humanos.head(10).itertuples(), 1):
        print(f"  {i:2d}. {row.termo:<30} +{row.distintividade:.4f}")

    print("\n" + "=" * 80)
    print("TOP 10 TERMOS MAIS DISTINTIVOS DE IA")
    print("=" * 80)
    for i, row in enumerate(ia.head(10).itertuples(), 1):
        print(f"  {i:2d}. {row.termo:<30} {row.distintividade:.4f}")

    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)

    return modelo, resultados


def analisar_sage_agregado(dirs_humano, dirs_ia, output_dir, nome_analise="agregado"):
    """
    Executa análise SAGE comparando múltiplos corpus agregados.

    Args:
        dirs_humano: lista de diretórios com textos humanos
        dirs_ia: lista de diretórios com textos de IA
        output_dir: diretório para salvar resultados
        nome_analise: nome da análise para identificação
    """
    print("=" * 80)
    print(f"ANÁLISE SAGE AGREGADA: {nome_analise}")
    print("=" * 80)

    # 1. Carregar todos os corpus
    print("\n[1/6] Carregando corpus de múltiplos diretórios...")

    todos_textos_humano = []
    todos_textos_ia = []

    for dir_humano in dirs_humano:
        dir_path = Path(dir_humano)
        if dir_path.exists():
            corpus = carregar_corpus(dir_humano)
            todos_textos_humano.extend(corpus.values())
            print(f"  Humano - {dir_path.name}: {len(corpus)} textos")
        else:
            print(f"  AVISO: Diretório não encontrado: {dir_humano}")

    for dir_ia in dirs_ia:
        dir_path = Path(dir_ia)
        if dir_path.exists():
            corpus = carregar_corpus(dir_ia)
            todos_textos_ia.extend(corpus.values())
            print(f"  IA - {dir_path.name}: {len(corpus)} textos")
        else:
            print(f"  AVISO: Diretório não encontrado: {dir_ia}")

    print(f"\n  Total textos humanos: {len(todos_textos_humano)}")
    print(f"  Total textos IA: {len(todos_textos_ia)}")

    if len(todos_textos_humano) == 0 or len(todos_textos_ia) == 0:
        print("\nERRO: Nenhum texto encontrado em um dos grupos!")
        return None, None

    # 2. Preparar dados e normalizar textos
    print("\n[2/7] Normalizando textos (lowercase, remover acentos)...")
    dados = []
    for texto in todos_textos_humano:
        dados.append((normalizar_texto(texto), "humano"))
    for texto in todos_textos_ia:
        dados.append((normalizar_texto(texto), "ia"))

    df = pd.DataFrame(dados, columns=["texto", "origem"])
    print(f"  Total de textos para análise: {len(df)}")
    print("  Normalização aplicada: lowercase + remoção de acentos")

    # 3. Vetorização
    print("\n[3/7] Vetorizando textos...")
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,  # Aumentado para corpus maior
        max_df=0.95,
        max_features=3000  # Aumentado para corpus maior
    )

    X = vectorizer.fit_transform(df["texto"])

    le = LabelEncoder()
    y = le.fit_transform(df["origem"])

    feature_names = vectorizer.get_feature_names_out()
    print(f"  Vocabulário: {len(feature_names)} termos")
    print(f"  Matriz esparsa: {X.shape}")

    # 4. Treinar SAGE
    print("\n[4/7] Treinando modelo SAGE...")
    modelo = SAGEModel(regularization=0.1)
    modelo.fit(X, y)
    print("  Modelo treinado com sucesso!")

    # 5. Extrair termos distintivos
    print("\n[5/7] Extraindo termos distintivos...")
    resultados = modelo.get_distinctive_terms(feature_names, top_k=50)

    # Ordenar por distintividade absoluta
    resultados['distintividade_abs'] = resultados['distintividade'].abs()
    resultados = resultados.sort_values('distintividade_abs', ascending=False)

    n_humano = len(resultados[resultados['classe'] == 'humano'])
    n_ia = len(resultados[resultados['classe'] == 'ia'])
    print(f"  Termos distintivos humanos: {n_humano}")
    print(f"  Termos distintivos IA: {n_ia}")

    # 6. Salvar resultados
    print("\n[6/7] Salvando resultados...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV com todos os termos
    csv_file = output_path / f"sage_termos_{nome_analise}.csv"
    resultados.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  CSV salvo: {csv_file}")

    # Relatório em texto
    report_file = output_path / f"sage_relatorio_{nome_analise}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RELATÓRIO SAGE AGREGADO - {nome_analise.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURAÇÃO:\n")
        f.write(f"  Diretórios Humano:\n")
        for d in dirs_humano:
            f.write(f"    - {d}\n")
        f.write(f"  Diretórios IA:\n")
        for d in dirs_ia:
            f.write(f"    - {d}\n")
        f.write(f"\n  Total textos humanos: {len(todos_textos_humano)}\n")
        f.write(f"  Total textos IA: {len(todos_textos_ia)}\n")
        f.write(f"  Total de textos: {len(df)}\n")
        f.write(f"  Vocabulário: {len(feature_names)} termos\n")
        f.write(f"  Regularização L1: {modelo.regularization}\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETAÇÃO DOS RESULTADOS:\n")
        f.write("=" * 80 + "\n")
        f.write("  - Distintividade POSITIVA: termo mais frequente em textos HUMANOS\n")
        f.write("  - Distintividade NEGATIVA: termo mais frequente em textos de IA\n")
        f.write("  - Quanto maior o valor absoluto, mais distintivo o termo\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 30 TERMOS MAIS DISTINTIVOS DE HUMANOS\n")
        f.write("=" * 80 + "\n\n")

        humanos = resultados[resultados['classe'] == 'humano'].head(30)
        for i, row in enumerate(humanos.itertuples(), 1):
            f.write(f"  {i:2d}. {row.termo:<30} +{row.distintividade:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 30 TERMOS MAIS DISTINTIVOS DE IA\n")
        f.write("=" * 80 + "\n\n")

        ia = resultados[resultados['classe'] == 'ia'].head(30)
        for i, row in enumerate(ia.itertuples(), 1):
            f.write(f"  {i:2d}. {row.termo:<30} {row.distintividade:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Relatório salvo: {report_file}")

    # 7. Imprimir resumo
    print("\n[7/7] Resumo dos resultados:")
    print("\n" + "=" * 80)
    print("TOP 15 TERMOS MAIS DISTINTIVOS DE HUMANOS")
    print("=" * 80)
    for i, row in enumerate(humanos.head(15).itertuples(), 1):
        print(f"  {i:2d}. {row.termo:<30} +{row.distintividade:.4f}")

    print("\n" + "=" * 80)
    print("TOP 15 TERMOS MAIS DISTINTIVOS DE IA")
    print("=" * 80)
    for i, row in enumerate(ia.head(15).itertuples(), 1):
        print(f"  {i:2d}. {row.termo:<30} {row.distintividade:.4f}")

    print("\n" + "=" * 80)
    print("ANÁLISE AGREGADA CONCLUÍDA!")
    print("=" * 80)

    return modelo, resultados


def main():
    """Função principal."""

    # Diretório base
    BASE_DIR = Path(__file__).parent.parent

    # Diretório de saída
    OUTPUT_DIR = Path(__file__).parent / "resultados"

    # =========================================================================
    # CONFIGURAÇÃO DOS CORPUS
    # =========================================================================

    # Diretórios de textos humanos
    dirs_humano = [
        BASE_DIR / 'corpus' / 'FakeTrue.Br-main' / 'fake',
        BASE_DIR / 'corpus' / 'Fake.br-Corpus-master' / 'full_texts' / 'fake_br_clean',
    ]

    # Diretórios de textos de IA
    dirs_ia = [
        BASE_DIR / 'corpus' / 'fake-news-llm-ptbr-main' / 'fake-news-llm-ptbr-main' / 'data' / 'FakeTrueBR',
        BASE_DIR / 'corpus' / 'fake-news-llm-ptbr-main' / 'fake-news-llm-ptbr-main' / 'data' / 'Fake.Br',
    ]

    # =========================================================================
    # EXECUTAR ANÁLISE AGREGADA (HUMANO vs MÁQUINA)
    # =========================================================================

    analisar_sage_agregado(
        dirs_humano=[str(d) for d in dirs_humano],
        dirs_ia=[str(d) for d in dirs_ia],
        output_dir=str(OUTPUT_DIR),
        nome_analise="Humano_vs_Maquina"
    )


if __name__ == '__main__':
    main()
