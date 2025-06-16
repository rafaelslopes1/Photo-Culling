# 🚀 Guia de Início Rápido - Photo Culling System 2.0

## Começando em 3 passos

### 1️⃣ Instalação
```bash
pip install -r requirements.txt
```

### 2️⃣ Verificação
```bash
python tools/health_check.py
```

### 3️⃣ Uso
```bash
# Interface web (recomendado)
python main.py --web-interface
# Acesse: http://localhost:5001

# Ou comandos específicos:
python main.py --extract-features
python main.py --train-model
python main.py --classify
```

---

## 📚 Próximos Passos

1. **Rotular imagens**: Use `--web-interface` para criar rótulos de treinamento
2. **Treinar IA**: Execute `--train-model` após ter alguns rótulos
3. **Classificar**: Use `--classify` para processar novas imagens automaticamente

## 🆘 Precisa de Ajuda?

- Consulte `README.md` para documentação completa
- Execute `python main.py --help` para todas as opções
- Verifique `docs/` para documentação detalhada
