"""
Configurações do sistema de análise de simulações
"""

# Configurações do Cassandra (HTC)
CASSANDRA_CONFIG = {
    'hosts': ['127.0.0.1'],
    'port': 9042,
    'keyspace': 'htc_reports',
    'table': 'simulation_reports'
}

# Configurações dos arquivos XML (Interscsimulator)
INTERSCSIMULATOR_CONFIG = {
    'data_dir': './data/interscsimulator',
    'file_pattern': '*.xml'
}

# Configurações de saída
OUTPUT_CONFIG = {
    'base_dir': './output',
    'reports_dir': './output/reports',
    'plots_dir': './output/plots',
    'comparison_dir': './output/comparison'
}

# Métricas a serem calculadas
METRICS_CONFIG = {
    'basic_metrics': [
        'total_vehicles',
        'total_distance',
        'average_speed',
        'travel_time',
        'link_density'
    ],
    'advanced_metrics': [
        'throughput',
        'congestion_index',
        'delay_time',
        'fuel_consumption',
        'emissions'
    ]
}

# Configurações de visualização
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'viridis',
    'heatmap_resolution': 100
}

# Configurações de comparação
COMPARISON_CONFIG = {
    'similarity_threshold': 0.8,
    'statistical_tests': ['t_test', 'ks_test', 'mann_whitney'],
    'correlation_methods': ['pearson', 'spearman', 'kendall']
}