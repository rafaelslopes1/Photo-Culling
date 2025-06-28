#!/bin/bash
# Photo Culling System - Automated Analysis Script
# Executa análise periódica das avaliações manuais

# Sair em caso de erro, tratar variáveis não definidas como erro e propagar o status de saída
set -euo pipefail

# Configurações
# Determina o diretório do projeto como o diretório pai do local do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/reports"
LOG_FILE="$PROJECT_DIR/logs/analysis.log"

# Criar diretórios se não existirem
mkdir -p "$REPORTS_DIR"
mkdir -p "$PROJECT_DIR/logs"

# Log com timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "Iniciando análise automática de avaliações"

# Navegar para o diretório do projeto
cd "$PROJECT_DIR" || {
    log_message "ERRO: Não foi possível acessar o diretório do projeto"
    exit 1
}

# Executar análise
python tools/evaluation_analyzer.py > "$REPORTS_DIR/latest_analysis.txt" 2>&1

if [ $? -eq 0 ]; then
    log_message "Análise executada com sucesso"
    
    # Mover relatório JSON para diretório de relatórios
    latest_json=$(ls -t evaluation_analysis_*.json 2>/dev/null | head -n1)
    if [ -n "$latest_json" ]; then
        mv "$latest_json" "$REPORTS_DIR/"
        log_message "Relatório JSON movido para $REPORTS_DIR/$latest_json"
    fi
    
    # Verificar se há alertas críticos
    if grep -q "Taxa de rejeição alta" "$REPORTS_DIR/latest_analysis.txt"; then
        log_message "ALERTA: Taxa de rejeição alta detectada"
        # Aqui poderia enviar email/notificação
    fi
    
    if grep -q "Amostra pequena" "$REPORTS_DIR/latest_analysis.txt"; then
        log_message "AVISO: Amostra ainda pequena para análises robustas"
    fi
    
else
    log_message "ERRO: Falha na execução da análise"
    exit 1
fi

log_message "Análise automática concluída"

# Limpeza de arquivos antigos (manter apenas últimos 30 dias)
find "$REPORTS_DIR" -name "evaluation_analysis_*.json" -mtime +30 -delete
find "$PROJECT_DIR/logs" -name "*.log" -mtime +60 -delete

log_message "Limpeza de arquivos antigos concluída"
