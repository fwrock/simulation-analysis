# Sistema de Análise de Simulações de Tráfego Urbano

Este sistema permite a análise e comparação entre simulações do **Hyperbolic Time Chamber (HTC)** e do **Interscsimulator**, dois simuladores mesoscópicos de tráfego urbano.

## Características Principais

### 🔍 Extração de Dados
- **HTC**: Extração direta do Cassandra (dados JSON)
- **Interscsimulator**: Parser de arquivos XML
- Normalização automática de IDs entre simuladores
- Suporte a múltiplas simulações simultâneas

### 📊 Métricas Calculadas
- **Básicas**: Total de veículos, distância percorrida, velocidade média, tempo de viagem
- **Tráfego**: Densidade de links, índice de congestionamento, variância de velocidade
- **Temporais**: Análise de métricas ao longo do tempo
- **Por Link**: Densidade, utilização de capacidade, throughput
- **Por Veículo**: Análise individual de jornadas

### 🔄 Sistema de Comparação
- Normalização de IDs (remove prefixos HTC: `htcaid_car_`, `htcaid_link_`)
- Mapeamento automático entre veículos equivalentes
- Testes estatísticos: t-test, Kolmogorov-Smirnov, Mann-Whitney
- Correlações: Pearson, Spearman, Kendall
- Score de similaridade e reprodutibilidade

### 📈 Visualizações
- Gráficos comparativos de métricas básicas
- Distribuição de velocidades
- Mapas de calor de densidade de links
- Análise temporal interativa
- Comparação de jornadas individuais
- Dashboard interativo com Plotly
- Relatórios HTML automáticos

## Estrutura do Projeto

```
simulations_analysis/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configurações gerais
├── src/
│   ├── __init__.py
│   ├── models.py                # Modelos de dados
│   ├── data_extraction/
│   │   ├── __init__.py
│   │   ├── htc_extractor.py     # Extrator Cassandra (HTC)
│   │   └── interscsimulator_extractor.py  # Extrator XML
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── calculator.py        # Calculador de métricas
│   ├── comparison/
│   │   ├── __init__.py
│   │   └── simulator_comparator.py  # Sistema de comparação
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py           # Sistema de visualização
├── output/                      # Saídas geradas
├── main.py                      # Script principal
├── requirements.txt
└── README.md
```

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Cassandra (HTC)

Certifique-se de que o Cassandra está rodando:

```bash
docker exec -it htc-cassandra-db cqlsh
```

### 3. Configurar Dados XML (Interscsimulator)

Coloque os arquivos XML no diretório configurado (padrão: `./data/interscsimulator/`)

## Uso

### Análise de Simulação Individual

```bash
# Analisar simulação HTC
python main.py --mode single --htc-sim "cenario_1000_viagens_2" --simulator htc

# Analisar simulação Interscsimulator
python main.py --mode single --ref-sim "data/interscsimulator/simulation_1.xml" --simulator interscsimulator
```

### Comparação entre Simulações

```bash
python main.py --mode compare \
    --htc-sim "cenario_1000_viagens_2" \
    --ref-sim "data/interscsimulator/simulation_1.xml"
```

### Análise de Jornada Individual

```bash
python main.py --mode vehicle \
    --htc-sim "cenario_1000_viagens_2" \
    --ref-sim "data/interscsimulator/simulation_1.xml" \
    --vehicle-id "trip_317"
```

### Análise em Lote

```bash
python main.py --mode batch \
    --htc-sims cenario_1000_viagens_1 cenario_1000_viagens_2 cenario_1000_viagens_3 \
    --ref-sims data/interscsimulator/sim1.xml data/interscsimulator/sim2.xml data/interscsimulator/sim3.xml
```

## Estrutura de Dados

### Eventos HTC (Cassandra)

```json
{
  "id": "uuid",
  "simulation_id": "cenario_1000_viagens_2",
  "car_id": "htcaid_car_trip_681",
  "event_type": "enter_link",
  "data": {
    "link_id": "htcaid_link_2085",
    "link_length": 38.08,
    "link_capacity": 600.0,
    "cars_in_link": 0,
    "calculated_speed": 4.17,
    "free_speed": 4.17,
    "travel_time": 7.19,
    "lanes": 1,
    "tick": 7060
  }
}
```

### Eventos Interscsimulator (XML)

```xml
<event time="155" type="enter_link" car_id="trip_1_1" 
       link_id="2067" link_length="16.12" link_capacity="2000" 
       cars_in_link="1" free_speed="8.33" calculated_speed="8.06" 
       travel_time="3" lanes="2" tick="155" />
```

## Normalização de IDs

O sistema automaticamente normaliza IDs entre simuladores:

### Padrões HTC
- Carros: `htcaid_car_trip_317` → `trip_317`
- Links: `htcaid_link_2114` → `2114`

### Padrões Interscsimulator
- Carros: `trip_317_1` → `trip_317`
- Links: mantém formato original

## Métricas Implementadas

### Métricas Básicas
- `total_vehicles`: Número total de veículos
- `total_distance`: Distância total percorrida (m)
- `average_speed`: Velocidade média (m/s)
- `average_travel_time`: Tempo médio de viagem (s)
- `simulation_duration`: Duração da simulação (s)
- `throughput`: Veículos por segundo

### Métricas de Tráfego
- `average_link_density`: Densidade média de links
- `max_link_density`: Densidade máxima observada
- `congestion_index`: Índice de congestionamento
- `speed_variance`: Variância da velocidade
- `delay_time`: Tempo de atraso médio

### Métricas de Comparação
- `similarity_score`: Score geral de similaridade (0-1)
- `reproducibility_score`: Score de reprodutibilidade (0-1)
- Correlações por métrica (Pearson, Spearman)
- Testes estatísticos com p-valores

## Saídas Geradas

### Arquivos JSON
- `analysis_{sim_id}_{simulator}.json`: Análise individual
- `comparison_{htc_id}_vs_{ref_id}.json`: Comparação entre simulações
- `vehicle_journey_{vehicle_id}.json`: Análise de jornada
- `batch_analysis_results.json`: Resultados de análise em lote

### Visualizações
- `basic_metrics_comparison.png`: Comparação de métricas básicas
- `speed_distribution.png`: Distribuição de velocidades
- `density_heatmap_*.png`: Mapas de calor de densidade
- `temporal_metrics_*.png`: Métricas temporais
- `link_comparison.png`: Comparação por link
- `vehicle_journey_*.png`: Jornadas individuais
- `interactive_dashboard.html`: Dashboard interativo
- `comparison_report.html`: Relatório final

## Configurações

Edite `config/settings.py` para personalizar:

### Cassandra
```python
CASSANDRA_CONFIG = {
    'hosts': ['127.0.0.1'],
    'port': 9042,
    'keyspace': 'htc_reports',
    'table': 'simulation_reports'
}
```

### Arquivos XML
```python
INTERSCSIMULATOR_CONFIG = {
    'data_dir': './data/interscsimulator',
    'file_pattern': '*.xml'
}
```

### Visualização
```python
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'viridis'
}
```

## Exemplos de Análise

### 1. Comparação de Performance

```python
from src.data_extraction.htc_extractor import HTCDataExtractor
from src.metrics.calculator import MetricsCalculator

# Extrair dados
with HTCDataExtractor() as extractor:
    events = extractor.get_events_by_simulation("cenario_1000_viagens_2")

# Calcular métricas
calculator = MetricsCalculator()
metrics = calculator.calculate_basic_metrics(events, "cenario_1000_viagens_2")

print(f"Throughput: {metrics.throughput:.3f} veículos/s")
print(f"Velocidade média: {metrics.average_speed:.2f} m/s")
```

### 2. Análise de Reprodutibilidade

```python
from src.comparison.simulator_comparator import SimulationComparator

comparator = SimulationComparator()
result = comparator.compare_simulations(htc_events, ref_events, "htc_sim", "ref_sim")

print(f"Similaridade: {result.similarity_score:.3f}")
print(f"Reprodutibilidade: {result.reproducibility_score:.3f}")
```

### 3. Análise Temporal

```python
temporal_data = calculator.calculate_time_series_metrics(events, time_window=300)
print(f"Pontos temporais: {len(temporal_data)}")
print(temporal_data[['time', 'unique_vehicles', 'avg_speed']].head())
```

## Troubleshooting

### Erro de Conexão Cassandra
```bash
# Verificar se container está rodando
docker ps | grep cassandra

# Verificar logs
docker logs htc-cassandra-db
```

### Arquivos XML Malformados
O sistema tenta múltiplas estratégias de parsing:
1. Parse XML completo
2. Parse streaming (iterparse)
3. Parse linha por linha

### Performance com Grandes Datasets
- Use `limit` na extração de eventos
- Analise em lotes menores
- Configure `time_window` adequado para análise temporal

## Desenvolvimento

### Extensões Futuras
- [ ] Suporte a mais formatos de dados
- [ ] Métricas de emissões e consumo
- [ ] Análise de rotas geoespaciais
- [ ] Comparação estatística avançada
- [ ] Interface web interativa
- [ ] Export para formatos acadêmicos

### Contribuição
1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente com testes
4. Abra um Pull Request

## Licença

Este projeto é desenvolvido para pesquisa acadêmica em simulação de tráfego urbano.

---

**Desenvolvido para análise comparativa entre simuladores HTC e Interscsimulator**# simulation-analysis
