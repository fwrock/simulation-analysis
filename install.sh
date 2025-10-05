#!/bin/bash

# Script de instalação e configuração do sistema de análise

echo "=== Sistema de Análise de Simulações de Tráfego Urbano ==="
echo "Instalando dependências e configurando ambiente..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Criar ambiente virtual (opcional)
read -p "Criar ambiente virtual? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Ambiente virtual criado e ativado"
fi

# Instalar dependências
echo "Instalando dependências Python..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependências instaladas com sucesso"
else
    echo "❌ Erro ao instalar dependências"
    exit 1
fi

# Criar diretórios necessários
echo "Criando diretórios de saída..."
mkdir -p output/reports
mkdir -p output/plots
mkdir -p output/comparison
mkdir -p data/interscsimulator

echo "✅ Diretórios criados"

# Verificar Cassandra (opcional)
echo "Verificando conexão com Cassandra..."
if command -v docker &> /dev/null; then
    if docker ps | grep -q cassandra; then
        echo "✅ Container Cassandra detectado"
        
        # Testar conexão
        echo "Testando conexão com Cassandra..."
        python3 -c "
from src.data_extraction.htc_extractor import HTCDataExtractor
try:
    extractor = HTCDataExtractor()
    if extractor.connect():
        print('✅ Conexão com Cassandra bem-sucedida')
        extractor.disconnect()
    else:
        print('❌ Falha na conexão com Cassandra')
except Exception as e:
    print(f'❌ Erro ao conectar Cassandra: {e}')
"
    else
        echo "⚠️  Container Cassandra não detectado"
        echo "   Para usar dados HTC, inicie o container Cassandra"
    fi
else
    echo "⚠️  Docker não encontrado"
fi

# Criar arquivo de configuração de exemplo
echo "Criando configuração de exemplo..."
cat > config/local_settings.py << 'EOF'
"""
Configurações locais (sobrescreve settings.py)
Copie e edite conforme necessário
"""

# Configurações do Cassandra para sua instalação
CASSANDRA_CONFIG = {
    'hosts': ['127.0.0.1'],  # Ajuste se necessário
    'port': 9042,
    'keyspace': 'htc_reports',
    'table': 'simulation_reports'
}

# Diretório dos seus arquivos XML
INTERSCSIMULATOR_CONFIG = {
    'data_dir': './data/interscsimulator',  # Ajuste conforme necessário
    'file_pattern': '*.xml'
}

# Configurações de saída
OUTPUT_CONFIG = {
    'base_dir': './output',
    'reports_dir': './output/reports',
    'plots_dir': './output/plots',
    'comparison_dir': './output/comparison'
}
EOF

echo "✅ Arquivo de configuração criado: config/local_settings.py"

# Executar exemplo
echo
read -p "Executar exemplo de uso? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executando exemplo..."
    python3 example_usage.py
fi

echo
echo "=== Instalação Concluída ==="
echo
echo "Próximos passos:"
echo "1. Configure seus dados em config/local_settings.py"
echo "2. Coloque arquivos XML em data/interscsimulator/"
echo "3. Execute: python3 main.py --help para ver opções"
echo
echo "Exemplos de uso:"
echo "  # Analisar simulação HTC"
echo "  python3 main.py --mode single --htc-sim 'cenario_1000_viagens_2'"
echo
echo "  # Comparar simulações"
echo "  python3 main.py --mode compare --htc-sim 'htc_sim' --ref-sim 'data/interscsimulator/ref_sim.xml'"
echo
echo "  # Análise em lote"
echo "  python3 main.py --mode batch --htc-sims sim1 sim2 --ref-sims data/interscsimulator/ref1.xml data/interscsimulator/ref2.xml"
echo