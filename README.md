# Caracterização lexical e sintática de notícias falsas em português produzidas por humanos e por máquinas

## Resumo

Notícias falsas são um grande problema para a sociedade. Com a Inteligência Artificial generativa, notícias falsas produzidas pela máquina têm se proliferado, tornando o cenário mais desafiador. Apesar da relevância desse problema, em línguas sub-representadas como o Português, as pesquisas que buscam diferenciar notícias falsas de humanos e de máquinas são incipientes. Buscando preencher essa lacuna, este artigo explora os corpora Fake.br e FakeTrueBR expandidos com notícias falsas geradas automaticamente, caracterizando lexical e sintaticamente as notícias falsas produzidas por humanos e por máquina. Os resultados mostram que textos gerados por máquina apresentam palavras significativamente mais longas, maior uso de modificadores adjetivais e menor diversidade sintática, apesar de utilizarem mais regras sintáticas por sentença. Em contrapartida, textos humanos exibem maior variabilidade estilística em todas as dimensões analisadas.

## Estrutura do repositório

- [corpus/](corpus/) — corpora utilizados (Fake.br, FakeTrueBR e a expansão com notícias falsas geradas por LLMs), além dos scripts de adaptação ([adapt_fake.py](corpus/adapt_fake.py), [adapt_llm_fake_files.py](corpus/adapt_llm_fake_files.py)).
- [zipf/](zipf/) — análise da distribuição de Zipf ([zipf.ipynb](zipf/zipf.ipynb)).
- [contagem_silabas/](contagem_silabas/) — contagem de sílabas por palavra e resultados agregados ([contagem_silabas.py](contagem_silabas/contagem_silabas.py)).
- [tagger/](tagger/) — etiquetagem morfossintática com Porttagger e análise agregada de POS humano vs. LLM ([pos_agregado_human_vs_llm.py](tagger/pos_agregado_human_vs_llm.py)).
- [parser/](parser/) — análise sintática com Portparser, extração de regras gramaticais, agregação, TF-IDF de regras e visualizações ([main.py](parser/main.py), [extract_grammar_rules.py](parser/extract_grammar_rules.py), [tfidf_rules.py](parser/tfidf_rules.py), [visualization.ipynb](parser/visualization.ipynb)).
- [sage/](sage/) — análises estatísticas com SAGE ([sage_analysis.py](sage/sage_analysis.py)).

## Recursos utilizados

- **Corpus base**: [fake-news-llm-ptbr](https://github.com/renatosvmor/fake-news-llm-ptbr)
- **Tagger**: [Porttagger](https://github.com/huberemanuel/porttagger)
- **Parser**: [Portparser](https://github.com/LuceleneL/Portparser)

## Citação (BibTeX)

Em breve.

## Agradecimentos

O presente trabalho foi realizado com apoio da Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES). Ele também contou com o apoio da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP; processo #2024/17834-6) e do Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq; processo #444933/2024-7).
