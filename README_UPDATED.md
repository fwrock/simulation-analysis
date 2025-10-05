# Sistema de Análise de Simulações de Tráfego

## 🎯 Visão Geral

Sistema Python avançado para análise comparativa entre simulações de tráfego urbano dos simuladores **HTC (Hyperbolic Time Chamber)** e **Interscsimulator**. O sistema oferece métricas estatísticas robustas, visualizações temporais adaptativas e relatórios metodológicos detalhados.

## 🚀 Características Principais

### Análise Comparativa Inteligente
- **Processamento Temporal Adaptativo**: Detecção automática do tipo de simulador com janelas temporais otimizadas
  - HTC: Janelas discretas para dados pontuais (ticks)
  - Interscsimulator: Janelas contínuas para timestamps
- **Normalização de IDs**: Mapeamento automático entre identificadores de veículos e links
- **Métricas Estatísticas**: Correlações de Pearson/Spearman, testes t, Kolmogorov-Smirnov, Mann-Whitney U

### Relatórios Metodológicos Detalhados
- **Descrições Científicas**: Explicações completas de como cada métrica é calculada
- **Interpretação Automática**: Avaliação qualitativa dos resultados (correlações, significância estatística)
- **Seções Expansíveis**: Interface HTML interativa com botões de expansão para metodologia
- **Considerações Técnicas**: Limitações, adaptações específicas e guias de interpretação

### Visualizações Avançadas
- **Gráficos Estáticos**: PNG com métricas básicas e distribuições
- **Dashboard Interativo**: Plotly com 6 gráficos comparativos
- **Séries Temporais Suavizadas**: Correção do problema de "zig-zag" em dados HTC

## 📊 Métricas Calculadas

### Scores Principais
- **Similaridade Geral** (0-1): Média de similaridade de veículos, eventos e temporal
- **Reprodutibilidade** (0-1): Sobreposição de veículos e similaridade de rotas

### Correlações
- **Pearson**: Relações lineares entre métricas temporais
- **Spearman**: Relações monotônicas (robusta a outliers)

### Testes Estatísticos
- **Teste t**: Comparação de médias de velocidade
- **Kolmogorov-Smirnov**: Comparação de distribuições
- **Mann-Whitney U**: Comparação não-paramétrica de medianas

### Diferenças Normalizadas
- Cálculo: |M_htc - M_ref| / max(|M_htc|, |M_ref|)
- Interpretação automática: Muito Similar → Muito Diferente

## 🛠️ Instalação e Configuração

### Pré-requisitos
```bash
Python 3.8+
Cassandra Database (para dados HTC)
```

### Instalação
```bash
# Clone o repositório
git clone <repository-url>
cd simulations_analysis

# Instalar dependências
bash install.sh
```

### Configuração
Edite `config/local_settings.py`:
```python
CASSANDRA_HOSTS = ['127.0.0.1']
CASSANDRA_PORT = 9042
CASSANDRA_KEYSPACE = 'htc_reports'
```

## 🎯 Uso do Sistema

### Comparação Básica
```bash
python main.py --mode compare \
  --htc-sim "simulation_id" \
  --ref-sim "/path/to/interscsimulator/events.xml"
```

### Análise de Veículo Específico
```bash
python main.py --mode vehicle \
  --htc-sim "simulation_id" \
  --vehicle-id "vehicle_123"
```

### Análise em Lote
```bash
python main.py --mode batch \
  --batch-file "batch_config.json"
```

## 📈 Interpretação dos Resultados

### Scores de Qualidade
- **Similaridade ≥ 0.8**: Simulações muito similares
- **Similaridade 0.6-0.8**: Simulações similares
- **Similaridade < 0.6**: Simulações divergentes

### Correlações
- **|r| ≥ 0.8**: Correlação muito forte
- **|r| 0.6-0.8**: Correlação forte
- **|r| 0.4-0.6**: Correlação moderada
- **|r| < 0.4**: Correlação fraca

### Significância Estatística
- **p < 0.001**: Diferença altamente significativa
- **p < 0.01**: Diferença muito significativa
- **p < 0.05**: Diferença significativa
- **p ≥ 0.05**: Sem diferença significativa

## 🔬 Metodologia Técnica

### Processamento Temporal Adaptativo
O sistema resolve o problema de visualização "zig-zag" dos dados HTC através de:

1. **Detecção de Simulador**: Análise automática dos atributos dos eventos
2. **Janelas Adaptativas baseadas na granularidade**: 
   - HTC: `max(1, duração/100)` ticks (granularidade fina)
   - Interscsimulator: `max(30, duração/50)` ticks (granularidade maior)
3. **Ordenação Temporal**: Múltiplos níveis de ordenação para garantir continuidade

### Normalização e Mapeamento
- **IDs de Veículos**: Normalização para permitir comparação direta
- **IDs de Links**: Mapeamento entre sistemas de numeração
- **Timestamps**: Alinhamento temporal baseado em eventos equivalentes

## 📁 Estrutura de Saída

```
output/
├── plots/
│   ├── basic_metrics_comparison.png      # Gráficos estáticos
│   ├── speed_distribution.png
│   ├── interactive_dashboard.html        # Dashboard interativo
│   └── comparison_report.html           # Relatório metodológico
├── reports/
│   └── detailed_analysis.json          # Dados brutos
└── comparison_results.json             # Resumo executivo
```

## 🔧 Solução de Problemas

### Erro de Conexão Cassandra
```bash
# Verificar status do Cassandra
sudo systemctl status cassandra

# Verificar conectividade
python debug_cassandra.py
```

### Dados Temporais "Zig-Zag"
✅ **Problema Resolvido**: O sistema agora usa janelas temporais adaptativas específicas para cada simulador.

### Performance em Datasets Grandes
- Use `--limit` para limitar número de eventos
- Considere análise em lote para múltiplas simulações
- Configure `BATCH_SIZE` em `config/settings.py`

## 📝 Exemplo de Relatório Metodológico

### Seções do Relatório HTML
1. **Resumo Executivo**: Scores principais com botões expandíveis
2. **Métricas de Correlação**: Valores com interpretação automática
3. **Diferenças Principais**: Avaliação qualitativa das diferenças
4. **Testes Estatísticos**: P-valores com interpretação de significância
5. **Considerações Metodológicas**: 
   - Adaptações específicas para simuladores
   - Limitações e guias de interpretação
6. **Gráficos**: Visualizações integradas

### Metodologias Explicadas
- **Score de Similaridade**: Componentes e fórmulas matemáticas
- **Correlações**: Diferenças entre Pearson e Spearman
- **Testes Estatísticos**: Hipóteses e interpretação de p-valores
- **Processamento Temporal**: Janelas adaptativas e sua necessidade

## 🤝 Contribuição

1. Fork o repositório
2. Crie branch para feature (`git checkout -b feature/nova-metrica`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova métrica'`)
4. Push para branch (`git push origin feature/nova-metrica`)
5. Crie Pull Request

## 📝 Changelog

### v2.1.0 (Atual)
- ✅ **Relatórios Metodológicos**: Descrições detalhadas de cada cálculo
- ✅ **Interface Expansível**: Botões de expansão para metodologia
- ✅ **Interpretação Automática**: Avaliação qualitativa de resultados
- ✅ **Correção Temporal**: Janelas adaptativas para HTC vs Interscsimulator

### v2.0.0
- ✅ **Processamento Adaptativo**: Detecção automática de simulador
- ✅ **Correção Zig-Zag**: Resolução de problemas de visualização temporal
- ✅ **Dashboard Interativo**: 6 gráficos comparativos com Plotly

### v1.0.0
- ✅ **Sistema Base**: Comparação básica entre simuladores
- ✅ **Métricas Estatísticas**: Correlações e testes de hipótese
- ✅ **Visualizações Estáticas**: Gráficos PNG comparativos

## 📄 Licença

MIT License - veja `LICENSE` para detalhes.

## 📞 Suporte

Para questões técnicas ou sugestões:
- Abra issue no repositório
- Consulte documentação em `docs/`
- Verifique logs em `simulation_analysis.log`