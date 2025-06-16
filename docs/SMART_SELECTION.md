# Sistema de Seleção Inteligente de Imagens

## Visão Geral

O Photo Culling System agora inclui um algoritmo de seleção inteligente que prioriza a rotulagem de imagens de classes sub-representadas, promovendo uma distribuição mais equilibrada dos dados de treinamento.

## Modos de Seleção

### 1. Seleção Sequencial (Padrão)
- **Modo**: `--selection-mode=sequential`
- **Comportamento**: Seleciona imagens em ordem alfabética determinística
- **Uso**: Adequado para rotulagem sistemática e reprodutível

### 2. Seleção Inteligente 
- **Modo**: `--selection-mode=smart`
- **Comportamento**: Prioriza imagens com maior probabilidade de pertencer a classes minoritárias
- **Uso**: Otimiza o processo de rotulagem para balanceamento de classes

## Algoritmo de Seleção Inteligente

### Critérios de Pontuação

O algoritmo combina três fatores para determinar a próxima imagem:

1. **Probabilidade de Classes Minoritárias (60% do peso)**
   - Identifica classes com menos de 50% da mediana de exemplos
   - Prioriza imagens com alta probabilidade de pertencer a essas classes

2. **Incerteza do Modelo (30% do peso)**
   - Seleciona imagens onde o modelo tem menor confiança
   - Maximiza o valor informativo para futuro treinamento

3. **Fator Aleatório (10% do peso)**
   - Adiciona diversidade na seleção
   - Evita loops de seleção previsíveis

### Condições de Fallback

- **IA Indisponível**: Usa seleção aleatória
- **Poucos Dados (< 10 rótulos)**: Usa seleção aleatória
- **Erro nas Predições**: Retorna à seleção sequencial

## Como Usar

### Através do Main.py

```bash
# Seleção sequencial (padrão)
python main.py --web-interface

# Seleção inteligente
python main.py --web-interface --selection-mode=smart

# Com porta personalizada
python main.py --web-interface --selection-mode=smart --port=5003
```

### Execução Direta

```bash
# Modo sequencial
python src/web/app.py

# Modo inteligente
python src/web/app.py --smart
```

## Logs e Transparência

O sistema fornece logs detalhados sobre as decisões de seleção:

```
🎯 SELEÇÃO INTELIGENTE: IMG_0123.JPG
   📊 Score final: 0.847
   🤖 Classe predita: quality_3 (45.2%)
   ⚖️  Classes sub-representadas: ['quality_1', 'reject_blur']
   📈 Score classes minoritárias: 0.832 (60% peso)
   ❓ Incerteza do modelo: 0.548 (30% peso)
   🎲 Fator aleatório: 10% peso
   🏆 Top 3 probabilidades: [('quality_3', 0.452), ('quality_2', 0.301), ('quality_4', 0.187)]
```

## Benefícios

### Para o Usuário
- **Eficiência**: Foca o esforço de rotulagem onde é mais necessário
- **Qualidade**: Melhora o balanceamento do dataset
- **Transparência**: Logs claros sobre as decisões do algoritmo

### Para o Modelo de IA
- **Dados Balanceados**: Reduz viés de classes majoritárias
- **Aprendizado Ativo**: Seleciona exemplos mais informativos
- **Performance**: Melhora a precisão geral do modelo

## Configuração

### Limites de Classes Minoritárias
- **Critério**: Classes com < 50% da mediana de exemplos
- **Mínimo**: Pelo menos 1 exemplo por classe
- **Dinâmico**: Recalculado a cada seleção

### Pesos dos Critérios
- **Classes Minoritárias**: 60%
- **Incerteza**: 30%
- **Aleatoriedade**: 10%

## Garantias

### Imagens Já Rotuladas
- ✅ **Nunca aparecem** na seleção automática
- ✅ Podem ser **revisadas manualmente** através da navegação
- ✅ Sistema **filtra automaticamente** imagens rotuladas

### Consistência
- ✅ **Ordem determinística** no modo sequencial
- ✅ **Logs completos** de todas as decisões
- ✅ **Fallbacks robustos** em caso de erro

## Exemplos de Uso

### Cenário 1: Dataset Novo
```bash
python main.py --web-interface --selection-mode=sequential
```
- Usar modo sequencial para primeiros rótulos
- Garante cobertura sistemática inicial

### Cenário 2: Balanceamento de Classes
```bash
python main.py --web-interface --selection-mode=smart
```
- Usar modo inteligente após ~50 rótulos
- Otimiza para classes sub-representadas

### Cenário 3: Revisão e Refinamento
```bash
python main.py --web-interface --selection-mode=smart --port=5003
```
- Usa seleção inteligente em porta específica
- Foca em exemplos mais informativos

## Status de Implementação

- ✅ **Seleção Inteligente**: Implementada e testada
- ✅ **Parâmetro de Linha de Comando**: Funcional
- ✅ **Logs Detalhados**: Implementados
- ✅ **Fallbacks**: Robustos
- ✅ **Garantia de Não-Repetição**: Implementada
- ✅ **Interface Web**: Compatível com ambos os modos

## Próximos Passos

1. **Métricas de Performance**: Comparar eficiência dos modos
2. **Configuração Avançada**: Permitir ajuste de pesos
3. **Visualização**: Dashboard de distribuição de classes
4. **Estratégias Híbridas**: Combinar múltiplas abordagens
