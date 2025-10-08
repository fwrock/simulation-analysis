"""
Sistema de comparação entre simulações HTC e Interscsimulator
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models import HTCEvent, InterscsimulatorEvent, ComparisonResult
from src.metrics.calculator import MetricsCalculator, BasicMetrics, TrafficMetrics
from src.data_extraction.htc_jsonl_extractor import HTCJsonlExtractor


@dataclass
class SimilarityMetrics:
    """Métricas de similaridade entre simulações"""
    overall_similarity: float
    speed_similarity: float
    density_similarity: float
    throughput_similarity: float
    spatial_similarity: float
    temporal_similarity: float


@dataclass
class ReproducibilityMetrics:
    """Métricas de reprodutibilidade"""
    vehicle_count_reproducibility: float
    route_reproducibility: float
    timing_reproducibility: float
    overall_reproducibility: float


class IDNormalizer:
    """Normaliza IDs entre os simuladores"""
    
    def __init__(self):
        # Padrões para IDs do HTC (com prefixos)
        self.htc_car_patterns = [
            r'htcaid_car_(.+)',      # htcaid_car_trip_317
            r'htcaid:car;(.+)',      # htcaid:car;trip_317
        ]
        
        self.htc_link_patterns = [
            r'htcaid_link_(.+)',     # htcaid_link_2114
            r'htcaid:link;(.+)',     # htcaid:link;2114
        ]
        
        # Padrões para IDs do simulador de referência
        self.ref_car_patterns = [
            r'(.+)_\d+$',            # trip_317_1, trip_317_2, etc.
            r'trip_(\d+)_\d+',       # trip_317_1 -> 317
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def normalize_car_id(self, car_id: str, simulator_type: str) -> str:
        """Normaliza ID de carro"""
        # Verificar se car_id é válido
        if car_id is None or (isinstance(car_id, float) and pd.isna(car_id)):
            return ""
        
        # Converter para string se necessário
        car_id = str(car_id)
        
        if simulator_type == 'htc':
            return self._normalize_htc_car_id(car_id)
        else:
            return self._normalize_ref_car_id(car_id)
    
    def normalize_link_id(self, link_id: str, simulator_type: str) -> str:
        """Normaliza ID de link"""
        # Verificar se link_id é válido
        if link_id is None or (isinstance(link_id, float) and pd.isna(link_id)):
            return ""
        
        # Converter para string se necessário
        link_id = str(link_id)
        
        if simulator_type == 'htc':
            return self._normalize_htc_link_id(link_id)
        else:
            return link_id  # Referência já está normalizada
    
    def _normalize_htc_car_id(self, car_id: str) -> str:
        """Remove prefixos do HTC de IDs de carros"""
        for pattern in self.htc_car_patterns:
            match = re.match(pattern, car_id)
            if match:
                return match.group(1)
        return car_id  # Retorna original se não houver match
    
    def _normalize_htc_link_id(self, link_id: str) -> str:
        """Remove prefixos do HTC de IDs de links"""
        for pattern in self.htc_link_patterns:
            match = re.match(pattern, link_id)
            if match:
                return match.group(1)
        return link_id  # Retorna original se não houver match
    
    def _normalize_ref_car_id(self, car_id: str) -> str:
        """Normaliza IDs do simulador de referência"""
        for pattern in self.ref_car_patterns:
            match = re.match(pattern, car_id)
            if match:
                return match.group(1)
        return car_id  # Retorna original se não houver match
    
    def create_mapping_table(self, htc_events: List[HTCEvent], 
                           ref_events: List[InterscsimulatorEvent]) -> Dict[str, str]:
        """Cria tabela de mapeamento entre IDs dos simuladores"""
        
        # Extrair IDs únicos normalizados
        htc_normalized = set()
        ref_normalized = set()
        
        for event in htc_events:
            normalized_car = self.normalize_car_id(event.car_id, 'htc')
            htc_normalized.add(normalized_car)
        
        for event in ref_events:
            normalized_car = self.normalize_car_id(event.car_id, 'interscsimulator')
            ref_normalized.add(normalized_car)
        
        # Criar mapeamento simples baseado em correspondência de nomes
        mapping = {}
        for htc_id in htc_normalized:
            if htc_id in ref_normalized:
                mapping[htc_id] = htc_id
        
        self.logger.info(f"Criado mapeamento para {len(mapping)} IDs comuns")
        return mapping


class SimulationComparator:
    """Comparador principal entre simulações"""
    
    def __init__(self):
        self.id_normalizer = IDNormalizer()
        self.metrics_calculator = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def compare_simulations(self, 
                          htc_events: List[HTCEvent], 
                          ref_events: List[InterscsimulatorEvent],
                          htc_sim_id: str,
                          ref_sim_id: str) -> ComparisonResult:
        """Compara duas simulações completas"""
        
        self.logger.info(f"🔄 Comparando simulação HTC {htc_sim_id} com {ref_sim_id}")
        self.logger.info(f"   📊 HTC: {len(htc_events):,} eventos")
        self.logger.info(f"   📊 Interscsimulator: {len(ref_events):,} eventos")
        
        # Normalizar IDs
        self.logger.info("🔄 Normalizando IDs dos eventos...")
        normalized_htc = self._normalize_events(htc_events, 'htc')
        normalized_ref = self._normalize_events(ref_events, 'interscsimulator')
        
        # Calcular métricas básicas
        self.logger.info("📊 Calculando métricas básicas HTC...")
        htc_metrics = self.metrics_calculator.calculate_basic_metrics(normalized_htc, htc_sim_id)
        
        self.logger.info("📊 Calculando métricas básicas Interscsimulator...")
        ref_metrics = self.metrics_calculator.calculate_basic_metrics(normalized_ref, ref_sim_id)
        
        # Calcular similaridade
        self.logger.info("🔍 Calculando similaridade geral...")
        similarity_score = self._calculate_overall_similarity(normalized_htc, normalized_ref)
        
        # Testes estatísticos
        self.logger.info("📈 Executando testes estatísticos...")
        statistical_tests = self._perform_statistical_tests(normalized_htc, normalized_ref)
        
        # Métricas de correlação
        self.logger.info("🔗 Calculando correlações...")
        correlation_metrics = self._calculate_correlations(normalized_htc, normalized_ref)
        
        # Diferenças específicas
        self.logger.info("📊 Comparando métricas...")
        differences = self.metrics_calculator.compare_metrics(htc_metrics, ref_metrics)
        
        # Reprodutibilidade
        self.logger.info("🎯 Calculando reprodutibilidade...")
        reproducibility_score = self._calculate_reproducibility(normalized_htc, normalized_ref)
        
        self.logger.info("✅ Comparação concluída!")
        
        return ComparisonResult(
            htc_simulation_id=htc_sim_id,
            interscsimulator_simulation_id=ref_sim_id,
            similarity_score=similarity_score,
            statistical_tests=statistical_tests,
            correlation_metrics=correlation_metrics,
            differences=differences,
            reproducibility_score=reproducibility_score
        )
    
    def compare_vehicle_journeys(self, 
                               htc_events: List[HTCEvent], 
                               ref_events: List[InterscsimulatorEvent],
                               vehicle_id: str) -> Dict[str, Any]:
        """Compara jornada de um veículo específico"""
        
        # Normalizar ID do veículo
        normalized_id = self.id_normalizer.normalize_car_id(vehicle_id, 'htc')
        
        # Filtrar eventos do veículo
        htc_vehicle = [e for e in htc_events if 
                      self.id_normalizer.normalize_car_id(e.car_id, 'htc') == normalized_id]
        ref_vehicle = [e for e in ref_events if 
                      self.id_normalizer.normalize_car_id(e.car_id, 'interscsimulator') == normalized_id]
        
        if not htc_vehicle or not ref_vehicle:
            return {'error': f'Veículo {vehicle_id} não encontrado em ambas simulações'}
        
        # Ordenar por timestamp
        htc_vehicle.sort(key=lambda x: x.timestamp)
        ref_vehicle.sort(key=lambda x: x.timestamp)
        
        comparison = {
            'vehicle_id': normalized_id,
            'htc_events_count': len(htc_vehicle),
            'ref_events_count': len(ref_vehicle),
            'htc_journey_time': htc_vehicle[-1].timestamp - htc_vehicle[0].timestamp,
            'ref_journey_time': ref_vehicle[-1].timestamp - ref_vehicle[0].timestamp,
        }
        
        # Comparar rotas (sequência de links)
        htc_route = self._extract_route(htc_vehicle, 'htc')
        ref_route = self._extract_route(ref_vehicle, 'interscsimulator')
        
        comparison.update({
            'htc_route_length': len(htc_route),
            'ref_route_length': len(ref_route),
            'route_similarity': self._calculate_route_similarity(htc_route, ref_route),
            'common_links': len(set(htc_route) & set(ref_route)),
            'htc_unique_links': len(set(htc_route) - set(ref_route)),
            'ref_unique_links': len(set(ref_route) - set(htc_route))
        })
        
        return comparison
    
    def calculate_link_density_comparison(self, 
                                        htc_events: List[HTCEvent], 
                                        ref_events: List[InterscsimulatorEvent]) -> pd.DataFrame:
        """Compara densidade de links entre simulações"""
        
        # Extrair dados de densidade
        htc_density = self._extract_link_density_data(htc_events, 'htc')
        ref_density = self._extract_link_density_data(ref_events, 'interscsimulator')
        
        # Normalizar IDs de links
        htc_density['normalized_link_id'] = htc_density['link_id'].apply(
            lambda x: self.id_normalizer.normalize_link_id(x, 'htc')
        )
        ref_density['normalized_link_id'] = ref_density['link_id'].apply(
            lambda x: self.id_normalizer.normalize_link_id(x, 'interscsimulator')
        )
        
        # Agrupar por link
        htc_grouped = htc_density.groupby('normalized_link_id').agg({
            'density': ['mean', 'max', 'std'],
            'calculated_speed': ['mean', 'std'],
            'cars_in_link': ['mean', 'max']
        }).reset_index()
        
        ref_grouped = ref_density.groupby('normalized_link_id').agg({
            'density': ['mean', 'max', 'std'],
            'calculated_speed': ['mean', 'std'],
            'cars_in_link': ['mean', 'max']
        }).reset_index()
        
        # Flatten column names
        htc_grouped.columns = ['link_id'] + [f'htc_{col[0]}_{col[1]}' for col in htc_grouped.columns[1:]]
        ref_grouped.columns = ['link_id'] + [f'ref_{col[0]}_{col[1]}' for col in ref_grouped.columns[1:]]
        
        # Merge comparisons
        comparison = pd.merge(htc_grouped, ref_grouped, on='link_id', how='inner')
        
        # Calculate differences
        comparison['density_diff'] = abs(comparison['htc_density_mean'] - comparison['ref_density_mean'])
        comparison['speed_diff'] = abs(comparison['htc_calculated_speed_mean'] - comparison['ref_calculated_speed_mean'])
        
        return comparison
    
    def generate_heatmap_comparison(self, 
                                  htc_events: List[HTCEvent], 
                                  ref_events: List[InterscsimulatorEvent],
                                  time_window: float = 300) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Gera dados para mapas de calor comparativos"""
        
        # Calcular métricas temporais para ambas simulações
        htc_temporal = self.metrics_calculator.calculate_time_series_metrics(htc_events, time_window)
        ref_temporal = self.metrics_calculator.calculate_time_series_metrics(ref_events, time_window)
        
        return htc_temporal, ref_temporal
    
    def _normalize_events(self, events: List[Any], simulator_type: str) -> List[Any]:
        """Normaliza IDs em eventos"""
        normalized_events = []
        
        for event in events:
            # Criar cópia do evento com IDs normalizados
            if hasattr(event, 'car_id'):
                normalized_car_id = self.id_normalizer.normalize_car_id(event.car_id, simulator_type)
                
                # Atualizar o evento com ID normalizado
                if hasattr(event, 'data') and 'link_id' in event.data:
                    normalized_link_id = self.id_normalizer.normalize_link_id(event.data['link_id'], simulator_type)
                    event.data['link_id'] = normalized_link_id
                elif hasattr(event, 'attributes') and 'link_id' in event.attributes:
                    normalized_link_id = self.id_normalizer.normalize_link_id(event.attributes['link_id'], simulator_type)
                    event.attributes['link_id'] = normalized_link_id
                
                # Atualizar car_id
                event.car_id = normalized_car_id
            
            normalized_events.append(event)
        
        return normalized_events
    
    def _calculate_overall_similarity(self, htc_events: List[Any], ref_events: List[Any]) -> float:
        """Calcula similaridade geral entre simulações"""
        
        # Converter para DataFrames
        htc_df = self.metrics_calculator._events_to_dataframe(htc_events)
        ref_df = self.metrics_calculator._events_to_dataframe(ref_events)
        
        if htc_df.empty or ref_df.empty:
            return 0.0
        
        similarities = []
        
        # Similaridade de contagem de veículos
        htc_vehicles = htc_df['car_id'].nunique()
        ref_vehicles = ref_df['car_id'].nunique()
        vehicle_similarity = 1 - abs(htc_vehicles - ref_vehicles) / max(htc_vehicles, ref_vehicles)
        similarities.append(vehicle_similarity)
        
        # Similaridade de eventos por tipo
        htc_events_by_type = htc_df['event_type'].value_counts(normalize=True)
        ref_events_by_type = ref_df['event_type'].value_counts(normalize=True)
        
        common_types = set(htc_events_by_type.index) & set(ref_events_by_type.index)
        if common_types:
            type_similarities = []
            for event_type in common_types:
                htc_prop = htc_events_by_type.get(event_type, 0)
                ref_prop = ref_events_by_type.get(event_type, 0)
                type_sim = 1 - abs(htc_prop - ref_prop)
                type_similarities.append(type_sim)
            similarities.append(np.mean(type_similarities))
        
        # Similaridade temporal
        htc_duration = htc_df['timestamp'].max() - htc_df['timestamp'].min()
        ref_duration = ref_df['timestamp'].max() - ref_df['timestamp'].min()
        if max(htc_duration, ref_duration) > 0:
            temporal_similarity = 1 - abs(htc_duration - ref_duration) / max(htc_duration, ref_duration)
            similarities.append(temporal_similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _perform_statistical_tests(self, htc_events: List[Any], ref_events: List[Any]) -> Dict[str, Any]:
        """Realiza testes estatísticos de comparação"""
        
        htc_df = self.metrics_calculator._events_to_dataframe(htc_events)
        ref_df = self.metrics_calculator._events_to_dataframe(ref_events)
        
        tests = {}
        
        if not htc_df.empty and not ref_df.empty:
            # Teste t para velocidades
            htc_speeds = htc_df[htc_df['event_type'] == 'enter_link']['calculated_speed'].dropna()
            ref_speeds = ref_df[ref_df['event_type'] == 'enter_link']['calculated_speed'].dropna()
            
            if len(htc_speeds) > 1 and len(ref_speeds) > 1:
                t_stat, t_p = stats.ttest_ind(htc_speeds, ref_speeds)
                tests['speed_t_test'] = {'statistic': t_stat, 'p_value': t_p}
                
                # Teste KS para distribuições
                ks_stat, ks_p = stats.ks_2samp(htc_speeds, ref_speeds)
                tests['speed_ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}
            
            # Teste para densidades
            htc_densities = htc_df[htc_df['event_type'] == 'enter_link']['cars_in_link'].dropna()
            ref_densities = ref_df[ref_df['event_type'] == 'enter_link']['cars_in_link'].dropna()
            
            if len(htc_densities) > 1 and len(ref_densities) > 1:
                mw_stat, mw_p = stats.mannwhitneyu(htc_densities, ref_densities, alternative='two-sided')
                tests['density_mannwhitney'] = {'statistic': mw_stat, 'p_value': mw_p}
        
        return tests
    
    def _calculate_correlations(self, htc_events: List[Any], ref_events: List[Any]) -> Dict[str, float]:
        """Calcula correlações entre métricas"""
        
        correlations = {}
        
        # Agregar métricas por janela de tempo
        htc_temporal = self.metrics_calculator.calculate_time_series_metrics(htc_events, 300)
        ref_temporal = self.metrics_calculator.calculate_time_series_metrics(ref_events, 300)
        
        if not htc_temporal.empty and not ref_temporal.empty:
            # Alinhar por tempo
            merged = pd.merge(htc_temporal, ref_temporal, on='time', suffixes=('_htc', '_ref'), how='inner')
            
            if not merged.empty:
                # Correlações para diferentes métricas
                for metric in ['avg_speed', 'avg_density', 'unique_vehicles']:
                    htc_col = f'{metric}_htc'
                    ref_col = f'{metric}_ref'
                    
                    if htc_col in merged.columns and ref_col in merged.columns:
                        htc_values = merged[htc_col].dropna()
                        ref_values = merged[ref_col].dropna()
                        
                        if len(htc_values) > 1 and len(ref_values) > 1:
                            # Correlação de Pearson
                            pearson_corr, _ = stats.pearsonr(htc_values, ref_values)
                            correlations[f'{metric}_pearson'] = pearson_corr
                            
                            # Correlação de Spearman
                            spearman_corr, _ = stats.spearmanr(htc_values, ref_values)
                            correlations[f'{metric}_spearman'] = spearman_corr
        
        return correlations
    
    def _calculate_reproducibility(self, htc_events: List[Any], ref_events: List[Any]) -> float:
        """Calcula score de reprodutibilidade"""
        
        htc_df = self.metrics_calculator._events_to_dataframe(htc_events)
        ref_df = self.metrics_calculator._events_to_dataframe(ref_events)
        
        if htc_df.empty or ref_df.empty:
            return 0.0
        
        reproducibility_scores = []
        
        # Reprodutibilidade de contagem de veículos
        htc_vehicles = set(htc_df['car_id'].unique())
        ref_vehicles = set(ref_df['car_id'].unique())
        vehicle_overlap = len(htc_vehicles & ref_vehicles) / len(htc_vehicles | ref_vehicles)
        reproducibility_scores.append(vehicle_overlap)
        
        # Reprodutibilidade de rotas (para veículos comuns)
        common_vehicles = htc_vehicles & ref_vehicles
        if common_vehicles:
            route_similarities = []
            for vehicle in list(common_vehicles)[:10]:  # Limitar a 10 veículos para performance
                htc_vehicle_events = htc_df[htc_df['car_id'] == vehicle]
                ref_vehicle_events = ref_df[ref_df['car_id'] == vehicle]
                
                htc_route = self._extract_route(htc_vehicle_events.to_dict('records'), 'htc')
                ref_route = self._extract_route(ref_vehicle_events.to_dict('records'), 'interscsimulator')
                
                route_sim = self._calculate_route_similarity(htc_route, ref_route)
                route_similarities.append(route_sim)
            
            if route_similarities:
                reproducibility_scores.append(np.mean(route_similarities))
        
        return np.mean(reproducibility_scores) if reproducibility_scores else 0.0
    
    def _extract_route(self, events: List[Any], simulator_type: str) -> List[str]:
        """Extrai sequência de links da jornada"""
        route = []
        
        for event in events:
            if hasattr(event, 'event_type') and event.event_type == 'enter_link':
                if hasattr(event, 'data') and 'link_id' in event.data:
                    link_id = self.id_normalizer.normalize_link_id(event.data['link_id'], simulator_type)
                    route.append(link_id)
                elif hasattr(event, 'attributes') and 'link_id' in event.attributes:
                    link_id = self.id_normalizer.normalize_link_id(event.attributes['link_id'], simulator_type)
                    route.append(link_id)
                elif isinstance(event, dict) and 'link_id' in event:
                    link_id = self.id_normalizer.normalize_link_id(event['link_id'], simulator_type)
                    route.append(link_id)
        
        return route
    
    def _calculate_route_similarity(self, route1: List[str], route2: List[str]) -> float:
        """Calcula similaridade entre duas rotas"""
        if not route1 or not route2:
            return 0.0
        
        # Similaridade baseada em sequência comum
        set1, set2 = set(route1), set(route2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Similaridade de sequência (considerando ordem)
        min_len = min(len(route1), len(route2))
        sequence_matches = sum(1 for i in range(min_len) if route1[i] == route2[i])
        sequence_similarity = sequence_matches / max(len(route1), len(route2))
        
        # Média ponderada
        return 0.7 * jaccard_similarity + 0.3 * sequence_similarity
    
    def _extract_link_density_data(self, events: List[Any], simulator_type: str) -> pd.DataFrame:
        """Extrai dados de densidade de links"""
        df = self.metrics_calculator._events_to_dataframe(events)
        enter_events = df[df['event_type'] == 'enter_link'].copy()
        
        if not enter_events.empty:
            enter_events['density'] = enter_events['cars_in_link'] / enter_events['link_capacity'].replace(0, 1)
        
        return enter_events
    
    def compare_with_jsonl_data(self, 
                               htc_jsonl_path: str,
                               ref_events: List[InterscsimulatorEvent],
                               ref_sim_id: str) -> Dict[str, Any]:
        """
        Compara dados do HTC via JSONL com eventos do simulador de referência
        
        Args:
            htc_jsonl_path: Caminho para o arquivo JSONL do HTC
            ref_events: Lista de eventos do simulador de referência
            ref_sim_id: ID da simulação de referência
            
        Returns:
            Dicionário com comparação detalhada incluindo análise de rotas
        """
        
        self.logger.info("🚀 Iniciando comparação com dados JSONL...")
        
        # Extrair dados do JSONL
        self.logger.info(f"📁 Carregando dados JSONL: {htc_jsonl_path}")
        jsonl_extractor = HTCJsonlExtractor(htc_jsonl_path)
        htc_df, htc_stats = jsonl_extractor.extract_events()
        
        if htc_df.empty:
            self.logger.error("❌ Nenhum evento encontrado no arquivo JSONL")
            return {}
        
        self.logger.info(f"✅ Carregados {len(htc_df):,} eventos do HTC")
        
        # Converter DataFrame em eventos para compatibilidade
        htc_events = self._dataframe_to_events(htc_df, 'htc')
        
        # Comparação padrão
        self.logger.info("🔄 Executando comparação padrão...")
        standard_comparison = self.compare_simulations(
            htc_events, ref_events, 
            htc_stats.get('simulation_id', 'htc_jsonl'), ref_sim_id
        )
        
        # Análise específica de rotas
        self.logger.info("🗺️ Analisando rotas...")
        routes_analysis = self._analyze_routes_comparison(htc_stats, ref_events)
        
        # Análise de eventos por tipo
        self.logger.info("📊 Analisando distribuição de eventos...")
        events_analysis = self._analyze_events_distribution(htc_stats, ref_events)
        
        # Análise temporal detalhada
        self.logger.info("⏱️ Analisando aspectos temporais...")
        temporal_analysis = self._analyze_temporal_aspects(htc_df, ref_events)
        
        # Compilar resultado
        result = {
            'standard_comparison': standard_comparison,
            'routes_analysis': routes_analysis,
            'events_analysis': events_analysis,
            'temporal_analysis': temporal_analysis,
            'htc_stats': htc_stats,
            'htc_events_count': len(htc_events),
            'ref_events_count': len(ref_events)
        }
        
        self.logger.info("✅ Comparação com JSONL concluída!")
        return result
    
    def _dataframe_to_events(self, df: pd.DataFrame, simulator_type: str) -> List:
        """Converte DataFrame em lista de eventos para compatibilidade"""
        events = []
        
        for _, row in df.iterrows():
            if simulator_type == 'htc':
                # Criar evento HTC com a estrutura correta
                event_data = {
                    'tick': row.get('tick', 0),
                    'event_type': row.get('data_event_type', 'unknown'),
                    'car_id': row.get('car_id', ''),
                    'link_id': row.get('link_id', ''),
                    'cars_in_link': row.get('cars_in_link', 0),
                    'link_capacity': row.get('link_capacity', 0),
                    'calculated_speed': row.get('calculated_speed', 0),
                    'travel_time': row.get('travel_time', 0),
                    'lanes': row.get('lanes', 0),
                    'total_distance': row.get('total_distance', 0),
                    'link_length': row.get('link_length', 0),
                    'free_speed': row.get('free_speed', 0)
                }
                
                event = HTCEvent(
                    timestamp=row.get('tick', 0),
                    car_id=row.get('car_id', ''),
                    event_type=row.get('data_event_type', 'unknown'),
                    tick=row.get('tick', 0),
                    simulation_id=row.get('simulation_id', ''),
                    node_id='',  # Não disponível nos dados JSONL
                    report_type='vehicle_flow',  # Tipo padrão
                    created_at=pd.Timestamp.now(),  # Timestamp atual
                    data=event_data
                )
            else:
                # Criar evento Interscsimulator
                event = InterscsimulatorEvent(
                    time=row.get('tick', 0),
                    event_type=row.get('data_event_type', 'unknown'),
                    car_id=row.get('car_id', ''),
                    link_id=row.get('link_id', ''),
                    cars_in_link=row.get('cars_in_link', 0),
                    link_capacity=row.get('link_capacity', 0),
                    calculated_speed=row.get('calculated_speed', 0)
                )
            
            events.append(event)
        
        return events
    
    def _analyze_routes_comparison(self, htc_stats: Dict, ref_events: List) -> Dict:
        """Analisa comparação de rotas entre simuladores"""
        
        # Extrair rotas do HTC
        htc_routes = htc_stats.get('routes_data', {})
        
        # Extrair rotas do simulador de referência
        ref_routes = {}
        ref_df = self.metrics_calculator._events_to_dataframe(ref_events)
        route_events = ref_df[ref_df['event_type'] == 'route_planned']
        
        for _, row in route_events.iterrows():
            car_id = row.get('car_id')
            if car_id:
                # Função auxiliar para processar strings que podem ser NaN/float
                def safe_split(value, delimiter=','):
                    if pd.isna(value) or not isinstance(value, str) or not value.strip():
                        return []
                    return value.split(delimiter)
                
                ref_routes[car_id] = {
                    'origin': row.get('origin'),
                    'destination': row.get('destination'),
                    'route_length': row.get('route_length'),
                    'route_cost': row.get('route_cost'),
                    'route_links': safe_split(row.get('route_links')),
                    'route_nodes': safe_split(row.get('route_nodes'))
                }
        
        # Comparar rotas
        routes_comparison = {
            'htc_routes_count': len(htc_routes),
            'ref_routes_count': len(ref_routes),
            'matching_vehicles': 0,
            'route_length_differences': [],
            'route_cost_differences': [],
            'route_complexity_differences': []
        }
        
        for car_id, htc_route in htc_routes.items():
            normalized_car_id = self.id_normalizer.normalize_car_id(car_id, 'htc')
            
            # Buscar rota correspondente
            ref_route = None
            for ref_car_id, ref_route_data in ref_routes.items():
                if normalized_car_id in ref_car_id or ref_car_id in normalized_car_id:
                    ref_route = ref_route_data
                    break
            
            if ref_route:
                routes_comparison['matching_vehicles'] += 1
                
                # Comparar comprimento
                htc_length = htc_route.get('route_length', 0)
                ref_length = ref_route.get('route_length', 0)
                if htc_length > 0 and ref_length > 0:
                    length_diff = abs(htc_length - ref_length) / max(htc_length, ref_length)
                    routes_comparison['route_length_differences'].append(length_diff)
                
                # Comparar custo
                htc_cost = htc_route.get('route_cost', 0)
                ref_cost = ref_route.get('route_cost', 0)
                if htc_cost > 0 and ref_cost > 0:
                    cost_diff = abs(htc_cost - ref_cost) / max(htc_cost, ref_cost)
                    routes_comparison['route_cost_differences'].append(cost_diff)
                
                # Comparar complexidade (número de links)
                htc_links = len(htc_route.get('route_links', []))
                ref_links = len(ref_route.get('route_links', []))
                if htc_links > 0 and ref_links > 0:
                    complexity_diff = abs(htc_links - ref_links) / max(htc_links, ref_links)
                    routes_comparison['route_complexity_differences'].append(complexity_diff)
        
        # Calcular estatísticas resumo
        if routes_comparison['route_length_differences']:
            routes_comparison['avg_length_difference'] = np.mean(routes_comparison['route_length_differences'])
            routes_comparison['max_length_difference'] = np.max(routes_comparison['route_length_differences'])
        
        if routes_comparison['route_cost_differences']:
            routes_comparison['avg_cost_difference'] = np.mean(routes_comparison['route_cost_differences'])
            routes_comparison['max_cost_difference'] = np.max(routes_comparison['route_cost_differences'])
        
        if routes_comparison['route_complexity_differences']:
            routes_comparison['avg_complexity_difference'] = np.mean(routes_comparison['route_complexity_differences'])
            routes_comparison['max_complexity_difference'] = np.max(routes_comparison['route_complexity_differences'])
        
        return routes_comparison
    
    def _analyze_events_distribution(self, htc_stats: Dict, ref_events: List) -> Dict:
        """Analisa distribuição de tipos de eventos"""
        
        htc_events_by_type = htc_stats.get('events_by_type', {})
        
        # Contar eventos do simulador de referência
        ref_events_by_type = {}
        for event in ref_events:
            event_type = event.event_type if hasattr(event, 'event_type') else getattr(event, 'type', 'unknown')
            ref_events_by_type[event_type] = ref_events_by_type.get(event_type, 0) + 1
        
        # Comparar distribuições
        all_event_types = set(htc_events_by_type.keys()) | set(ref_events_by_type.keys())
        
        comparison = {
            'htc_total': sum(htc_events_by_type.values()),
            'ref_total': sum(ref_events_by_type.values()),
            'event_type_comparison': {},
            'missing_in_htc': [],
            'missing_in_ref': [],
            'distribution_differences': {}
        }
        
        for event_type in all_event_types:
            htc_count = htc_events_by_type.get(event_type, 0)
            ref_count = ref_events_by_type.get(event_type, 0)
            
            comparison['event_type_comparison'][event_type] = {
                'htc_count': htc_count,
                'ref_count': ref_count,
                'difference': htc_count - ref_count,
                'htc_percentage': htc_count / comparison['htc_total'] * 100 if comparison['htc_total'] > 0 else 0,
                'ref_percentage': ref_count / comparison['ref_total'] * 100 if comparison['ref_total'] > 0 else 0
            }
            
            if htc_count == 0 and ref_count > 0:
                comparison['missing_in_htc'].append(event_type)
            elif ref_count == 0 and htc_count > 0:
                comparison['missing_in_ref'].append(event_type)
            
            # Calcular diferença de distribuição
            if comparison['htc_total'] > 0 and comparison['ref_total'] > 0:
                htc_prop = htc_count / comparison['htc_total']
                ref_prop = ref_count / comparison['ref_total']
                comparison['distribution_differences'][event_type] = abs(htc_prop - ref_prop)
        
        return comparison
    
    def _analyze_temporal_aspects(self, htc_df: pd.DataFrame, ref_events: List) -> Dict:
        """Analisa aspectos temporais das simulações"""
        
        # Duração das simulações
        htc_duration = htc_df['tick'].max() - htc_df['tick'].min() if not htc_df.empty else 0
        
        ref_times = [getattr(event, 'time', getattr(event, 'timestamp', 0)) for event in ref_events]
        ref_duration = max(ref_times) - min(ref_times) if ref_times else 0
        
        # Taxa de eventos por tick
        htc_events_per_tick = len(htc_df) / htc_duration if htc_duration > 0 else 0
        ref_events_per_tick = len(ref_events) / ref_duration if ref_duration > 0 else 0
        
        # Análise de picos de atividade
        htc_events_by_tick = htc_df.groupby('tick').size() if not htc_df.empty else pd.Series()
        
        ref_df = self.metrics_calculator._events_to_dataframe(ref_events)
        # Corrigir nome da coluna - usar 'tick' em vez de 'time'
        ref_events_by_tick = ref_df.groupby('tick').size() if not ref_df.empty else pd.Series()
        
        analysis = {
            'htc_duration': htc_duration,
            'ref_duration': ref_duration,
            'duration_difference': abs(htc_duration - ref_duration),
            'htc_events_per_tick': htc_events_per_tick,
            'ref_events_per_tick': ref_events_per_tick,
            'events_rate_difference': abs(htc_events_per_tick - ref_events_per_tick),
            'htc_peak_activity': htc_events_by_tick.max() if not htc_events_by_tick.empty else 0,
            'ref_peak_activity': ref_events_by_tick.max() if not ref_events_by_tick.empty else 0,
            'htc_avg_activity': htc_events_by_tick.mean() if not htc_events_by_tick.empty else 0,
            'ref_avg_activity': ref_events_by_tick.mean() if not ref_events_by_tick.empty else 0
        }
        
        return analysis


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    comparator = SimulationComparator()
    print("Sistema de comparação inicializado")