"""
Gera um requirements.txt a partir de UM script especifico.

Como usar:
    python gerar_requirements.py servidor.py

Observacao: os pacotes precisam estar instalados no ambiente atual,
porque e dele que o script le os nomes e as versoes.
"""

import ast
import os
import sys
from importlib.metadata import packages_distributions, version


def listar_imports(caminho):
    """Le o script e devolve os nomes de tudo que ele importa."""
    with open(caminho, encoding="utf-8") as arquivo:
        # ast transforma o codigo em uma "arvore" que da para percorrer
        arvore = ast.parse(arquivo.read())

    modulos = set()
    for no in ast.walk(arvore):
        # Caso: import numpy   /   import os.path
        if isinstance(no, ast.Import):
            for item in no.names:
                modulos.add(item.name.split(".")[0])

        # Caso: from PIL import Image
        # (level == 0 ignora imports relativos, como "from . import x")
        elif isinstance(no, ast.ImportFrom):
            if no.level == 0 and no.module:
                modulos.add(no.module.split(".")[0])
    return modulos


def main():
    if len(sys.argv) < 2:
        print("Uso: python gerar_requirements.py nome_do_script.py")
        return

    caminho = sys.argv[1]
    pasta = os.path.dirname(os.path.abspath(caminho))

    # Dicionario que traduz "nome do import" para "nome do pacote"
    # Exemplo: {"PIL": ["pillow"], "numpy": ["numpy"]}
    traducao = packages_distributions()

    linhas = set()
    for modulo in sorted(listar_imports(caminho)):

        # 1) Bibliotecas que ja vem com o Python (os, io, re, base64...)
        if modulo in sys.stdlib_module_names:
            continue

        # 2) Arquivos seus, na mesma pasta (ex: from servidor import app)
        if os.path.exists(os.path.join(pasta, modulo + ".py")):
            print(f"Ignorado (arquivo local): {modulo}")
            continue

        # 3) Pacotes de terceiros: descobre o nome real e a versao instalada
        if modulo in traducao:
            for pacote in traducao[modulo]:
                linhas.add(f"{pacote}=={version(pacote)}")
        else:
            print(f"Aviso: '{modulo}' nao esta instalado. Rode: pip install {modulo}")

    with open("requirements.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(sorted(linhas)) + "\n")

    print("\nrequirements.txt criado com:")
    for linha in sorted(linhas):
        print(" ", linha)


main()
