# Pipe

Acesse o Jupyter Lab no seu navegador clicando ou acessando o link abaixo:
**[http://localhost:8888/](http://localhost:8888/)**

Os notebooks ficam em [`notebooks/`](notebooks/) e devem ser executados **nesta ordem**:

1. `Extract.ipynb` — lê os dados brutos e gera a Camada Bronze
2. `Transform.ipynb` — limpa, cruza as bases e gera a Camada Ouro
3. `Load.ipynb` — carrega a Camada Ouro no PostgreSQL (veja [`../Postgress/`](../Postgress/))
4. `Model.py` — modelagem, testes estatísticos e exportação dos resultados finais

Detalhes da pipeline, das fontes de dados e da metodologia estão no
[README principal do projeto](../README.md).
