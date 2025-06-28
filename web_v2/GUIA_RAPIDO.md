# 🚀 Guia Rápido - Photo Culling System

## ⚡ Início Imediato (2 minutos)

### 1. Iniciar Sistema
```bash
cd /Users/rafaellopes/www/Fotop/Photo-Culling/web_v2
python backend/app_simple.py
```

### 2. Acessar Interface
- **Avaliação**: http://localhost:5001
- **Dashboard**: http://localhost:5001/dashboard

### 3. Primeira Avaliação
1. ✅ **Veja a imagem** - use zoom/pan se necessário
2. ✅ **Avalie qualidade geral** (1-5 estrelas)
3. ✅ **Defina contexto** (pessoas, iluminação)
4. ✅ **Marque aprovações** (portfólio/cliente/social)
5. ✅ **Indique confiança** (deslizador)
6. ✅ **Clique "Próxima Imagem"**

---

## 📊 Análise de Dados (1 minuto)

### Relatório Automático
```bash
python tools/evaluation_analyzer.py
```

### Dashboard Visual
- Acesse: http://localhost:5001/dashboard
- Veja: estatísticas em tempo real
- Monitore: progresso e tendências

---

## 🎯 Metas Diárias

### Para Avaliadores
- **5-10 avaliações/dia** = progresso consistente
- **>85% confiança** = padrões claros
- **Variar contextos** = dados equilibrados

### Para Gestores
- **Check dashboard** = monitorar progresso
- **Review semanal** = ajustar estratégias
- **Análise mensal** = otimizar processo

---

## 🚨 Alertas Importantes

### ⚠️ Taxa de Rejeição Alta (>70%)
- **Ação**: Implementar filtros automáticos básicos
- **Verificar**: Critérios de pré-seleção de imagens

### ⚠️ Amostra Pequena (<25)
- **Ação**: Acelerar coleta de avaliações
- **Meta**: 25+ para análises confiáveis

### ⚠️ Baixa Confiança (<80%)
- **Ação**: Revisar guidelines de avaliação
- **Treinar**: Critérios de qualidade

---

## 🎉 Milestones do Projeto

| Avaliações | Status | Próximas Ações |
|------------|--------|----------------|
| **0-10** | 🔄 Início | Estabelecer rotina |
| **10-25** | 📈 Crescimento | Acelerar coleta |
| **25-50** | 📊 Análises básicas | Correlações |
| **50-100** | 🤖 ML básico | Modelos iniciais |
| **100+** | 🚀 Produção | Sistema híbrido |

---

## 💡 Dicas de Eficiência

### Interface
- **Teclas rápidas**: Enter = próxima, Esc = reset
- **Zoom inteligente**: Duplo clique na imagem
- **Navegação fluida**: Use os controles otimizados

### Avaliação
- **Primeiro olhar**: Impressão geral (1 segundo)
- **Análise técnica**: Zoom em detalhes críticos
- **Decisão final**: Baseada em critérios claros

### Dados
- **Seja consistente**: Use mesmos critérios sempre
- **Seja honesto**: Indique confiança real
- **Seja específico**: Use comentários quando necessário

---

## 🔄 Automação Disponível

### Script Diário
```bash
# Adicionar ao crontab para 18h diárias
0 18 * * * /path/to/tools/automated_analysis.sh
```

### Relatórios Periódicos
- **Diários**: Análise de progresso
- **Semanais**: Tendências e insights
- **Mensais**: Otimizações e ajustes

---

**⚡ Lembre-se**: O sistema está otimizado para velocidade e precisão. Foque na qualidade das avaliações, o sistema cuida da análise e insights automaticamente!
