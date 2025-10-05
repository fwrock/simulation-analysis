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
