import os
import unicodedata
from pathlib import Path

PASTA = Path("assets/escudos")

def normalizar(nome):
    nome = nome.lower().strip()
    nome = unicodedata.normalize('NFKD', nome)\
           .encode('ASCII', 'ignore')\
           .decode('ASCII')
    nome = nome.replace(" ", "_")
    nome = nome.replace("-", "_")

    permitido = "abcdefghijklmnopqrstuvwxyz0123456789_"
    nome = "".join(c for c in nome if c in permitido)

    return nome

for arquivo in PASTA.glob("*.png"):

    if arquivo.name.lower() == "default.png":
        continue

    novo_nome = normalizar(arquivo.stem) + ".png"
    novo_caminho = arquivo.with_name(novo_nome)

    if not novo_caminho.exists():
        os.rename(arquivo, novo_caminho)
        print(f"{arquivo.name} → {novo_nome}")

print("✔ Escudos padronizados")
