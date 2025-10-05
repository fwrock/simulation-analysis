"""
Calculador de métricas para análise de simulações de tráfego
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.models import HTCEvent, InterscsimulatorEvent


@dataclass
class BasicMetrics:
    """Métricas básicas de uma simulação"""
    total_vehicles: int
    total_distance: float
    average_speed: float
    average_travel_time: float
    simulation_duration: float
    throughput: float  # veículos por unidade de tempo
    

@dataclass
class TrafficMetrics:
    """Métricas específicas de tráfego"""
    average_link_density: float
    max_link_density: float
    congestion_index: float  # densidade média / capacidade média
    speed_variance: float
    delay_time: float  # diferença entre tempo real e tempo livre
    

@dataclass
class LinkMetrics:
    """Métricas por link"""
    link_id: str
    average_density: float
    max_density: float
    average_speed: float
    capacity_utilization: float
    throughput: float
    congestion_duration: float  # tempo em congestionamento


class MetricsCalculator:
    """Calculador de métricas para simulações de tráfego"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, events: List[Any], simulation_id: str) -> BasicMetrics:
        """Calcula métricas básicas de uma simulação"""
        
        if not events:
            return BasicMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Converter para DataFrame para facilitar análise
        df = self._events_to_dataframe(events)
        
        # Veículos únicos
        unique_vehicles = df['car_id'].nunique()
        
        # Duração da simulação
        if len(df) > 0:
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            duration = end_time - start_time
        else:
            duration = 0
        
        # Distância total (do último evento de cada veículo)
        total_distance = 0
        for car_id in df['car_id'].unique():
            car_events = df[df['car_id'] == car_id]
            # Buscar eventos de completed ou leave_link com total_distance
            completed_events = car_events[car_events['event_type'] == 'journey_completed']
            if not completed_events.empty:
                total_distance += completed_events['total_distance'].iloc[-1]
            else:
                # Tentar leave_link events
                leave_events = car_events[car_events['event_type'] == 'leave_link']
                if not leave_events.empty:
                    total_distance += leave_events['total_distance'].max()
        
        # Velocidade média (dos eventos enter_link)
        enter_events = df[df['event_type'] == 'enter_link']
        avg_speed = enter_events['calculated_speed'].mean() if not enter_events.empty else 0
        
        # Tempo de viagem médio
        travel_times = []
        for car_id in df['car_id'].unique():
            car_events = df[df['car_id'] == car_id].sort_values('timestamp')
            if len(car_events) >= 2:
                start = car_events['timestamp'].iloc[0]
                end = car_events['timestamp'].iloc[-1]
                travel_times.append(end - start)
        
        avg_travel_time = np.mean(travel_times) if travel_times else 0
        
        # Throughput
        throughput = unique_vehicles / duration if duration > 0 else 0
        
        return BasicMetrics(
            total_vehicles=unique_vehicles,
            total_distance=total_distance,
            average_speed=avg_speed,
            average_travel_time=avg_travel_time,
            simulation_duration=duration,
            throughput=throughput
        )
    
    def calculate_traffic_metrics(self, events: List[Any]) -> TrafficMetrics:
        """Calcula métricas específicas de tráfego"""
        
        df = self._events_to_dataframe(events)
        enter_events = df[df['event_type'] == 'enter_link'].copy()
        
        if enter_events.empty:
            return TrafficMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calcular densidade (cars_in_link / link_capacity)
        enter_events['density'] = enter_events['cars_in_link'] / enter_events['link_capacity'].replace(0, 1)
        
        # Métricas de densidade
        avg_density = enter_events['density'].mean()
        max_density = enter_events['density'].max()
        
        # Índice de congestionamento
        congestion_index = avg_density  # Densidade média como índice de congestionamento
        
        # Variância da velocidade
        speed_variance = enter_events['calculated_speed'].var()
        
        # Tempo de atraso (diferença entre velocidade livre e calculada)
        enter_events['delay_factor'] = (enter_events['free_speed'] - enter_events['calculated_speed']) / enter_events['free_speed']
        enter_events['delay_factor'] = enter_events['delay_factor'].clip(lower=0)
        avg_delay = enter_events['delay_factor'].mean() * enter_events['travel_time'].mean()
        
        return TrafficMetrics(
            average_link_density=avg_density,
            max_link_density=max_density,
            congestion_index=congestion_index,
            speed_variance=speed_variance,
            delay_time=avg_delay
        )
    
    def calculate_link_metrics(self, events: List[Any]) -> List[LinkMetrics]:
        """Calcula métricas por link"""
        
        df = self._events_to_dataframe(events)
        enter_events = df[df['event_type'] == 'enter_link'].copy()
        
        if enter_events.empty:
            return []
        
        link_metrics = []
        
        for link_id in enter_events['link_id'].unique():
            link_events = enter_events[enter_events['link_id'] == link_id]
            
            # Calcular densidade
            link_events['density'] = link_events['cars_in_link'] / link_events['link_capacity'].replace(0, 1)
            
            # Métricas do link
            avg_density = link_events['density'].mean()
            max_density = link_events['density'].max()
            avg_speed = link_events['calculated_speed'].mean()
            capacity_utilization = link_events['cars_in_link'].max() / link_events['link_capacity'].iloc[0]
            
            # Throughput do link (veículos únicos que passaram)
            unique_vehicles = link_events['car_id'].nunique()
            duration = link_events['timestamp'].max() - link_events['timestamp'].min()
            throughput = unique_vehicles / duration if duration > 0 else 0
            
            # Duração de congestionamento (densidade > 0.8)
            congested_events = link_events[link_events['density'] > 0.8]
            congestion_duration = len(congested_events) / len(link_events) if len(link_events) > 0 else 0
            
            link_metrics.append(LinkMetrics(
                link_id=link_id,
                average_density=avg_density,
                max_density=max_density,
                average_speed=avg_speed,
                capacity_utilization=capacity_utilization,
                throughput=throughput,
                congestion_duration=congestion_duration
            ))
        
        return link_metrics
    
    def calculate_vehicle_metrics(self, events: List[Any], car_id: str) -> Dict[str, float]:
        """Calcula métricas para um veículo específico"""
        
        df = self._events_to_dataframe(events)
        vehicle_events = df[df['car_id'] == car_id].sort_values('timestamp')
        
        if vehicle_events.empty:
            return {}
        
        metrics = {}
        
        # Tempo total de viagem
        if len(vehicle_events) >= 2:
            metrics['total_travel_time'] = vehicle_events['timestamp'].iloc[-1] - vehicle_events['timestamp'].iloc[0]
        
        # Distância total
        completed_events = vehicle_events[vehicle_events['event_type'] == 'journey_completed']
        if not completed_events.empty:
            metrics['total_distance'] = completed_events['total_distance'].iloc[-1]
        
        # Velocidade média
        enter_events = vehicle_events[vehicle_events['event_type'] == 'enter_link']
        if not enter_events.empty:
            metrics['average_speed'] = enter_events['calculated_speed'].mean()
            metrics['max_speed'] = enter_events['calculated_speed'].max()
            metrics['min_speed'] = enter_events['calculated_speed'].min()
        
        # Número de links visitados
        metrics['links_visited'] = enter_events['link_id'].nunique() if not enter_events.empty else 0
        
        # Tempo em congestionamento (densidade > 0.8)
        if not enter_events.empty:
            enter_events['density'] = enter_events['cars_in_link'] / enter_events['link_capacity'].replace(0, 1)
            congested_time = enter_events[enter_events['density'] > 0.8]['travel_time'].sum()
            metrics['congestion_time'] = congested_time
            metrics['congestion_ratio'] = congested_time / enter_events['travel_time'].sum()
        
        return metrics
    
    def _detect_simulator_type(self, events: List[Any]) -> str:
        """Detecta o tipo de simulador baseado nos eventos"""
        if not events:
            return "unknown"
        
        # Verificar tipo da primeira instância
        first_event = events[0]
        if hasattr(first_event, 'data'):
            return "htc"
        elif hasattr(first_event, 'attributes'):
            return "interscsimulator"
        else:
            return "unknown"
    
    def _get_optimal_time_window(self, df: pd.DataFrame, simulator_type: str) -> float:
        """Calcula janela de tempo otimizada baseado no tipo de simulador"""
        time_range = df['timestamp'].max() - df['timestamp'].min()
        
        if simulator_type == "htc":
            # Para HTC (ticks discretos), usar janelas menores
            return max(1, time_range / 100)  # 100 janelas para boa resolução
        elif simulator_type == "interscsimulator":
            # Para Interscsimulator (tempo contínuo), usar janelas maiores  
            return max(30, time_range / 50)   # 50 janelas para suavizar
        else:
            # Auto-detectar baseado no range
            if time_range < 10000:
                return max(10, time_range / 50)
            else:
                return max(100, time_range / 30)

    def calculate_time_series_metrics(self, events: List[Any], time_window: float = None) -> pd.DataFrame:
        """Calcula métricas em janelas de tempo"""
        
        df = self._events_to_dataframe(events)
        
        if df.empty:
            return pd.DataFrame()
        
        # Detectar tipo de simulador
        simulator_type = self._detect_simulator_type(events)
        
        # Usar janela de tempo otimizada se não especificada
        if time_window is None:
            time_window = self._get_optimal_time_window(df, simulator_type)
        
        # Ordenar dados por timestamp primeiro
        df = df.sort_values('timestamp')
        
        # Criar janelas de tempo
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        time_bins = np.arange(start_time, end_time + time_window, time_window)
        
        df['time_bin'] = pd.cut(df['timestamp'], bins=time_bins, labels=time_bins[:-1])
        
        # Calcular métricas por janela de tempo
        time_metrics = []
        
        # Ordenar os bins para garantir sequência temporal
        for time_bin in sorted(df['time_bin'].dropna().unique()):
            window_events = df[df['time_bin'] == time_bin]
            
            # Métricas básicas da janela
            metrics = {
                'time': time_bin,
                'total_events': len(window_events),
                'unique_vehicles': window_events['car_id'].nunique(),
                'journey_started': len(window_events[window_events['event_type'] == 'journey_started']),
                'journey_completed': len(window_events[window_events['event_type'] == 'journey_completed']),
                'enter_link': len(window_events[window_events['event_type'] == 'enter_link']),
                'leave_link': len(window_events[window_events['event_type'] == 'leave_link'])
            }
            
            # Métricas de tráfego da janela
            enter_events = window_events[window_events['event_type'] == 'enter_link']
            if not enter_events.empty:
                enter_events['density'] = enter_events['cars_in_link'] / enter_events['link_capacity'].replace(0, 1)
                metrics.update({
                    'avg_density': enter_events['density'].mean(),
                    'avg_speed': enter_events['calculated_speed'].mean(),
                    'avg_cars_in_link': enter_events['cars_in_link'].mean()
                })
            else:
                metrics.update({
                    'avg_density': 0,
                    'avg_speed': 0,
                    'avg_cars_in_link': 0
                })
            
            time_metrics.append(metrics)
        
        # Converter para DataFrame e ordenar por tempo
        result_df = pd.DataFrame(time_metrics)
        if not result_df.empty:
            result_df = result_df.sort_values('time').reset_index(drop=True)
        
        return result_df
    
    def compare_metrics(self, metrics1: BasicMetrics, metrics2: BasicMetrics) -> Dict[str, float]:
        """Compara métricas entre duas simulações"""
        
        comparison = {}
        
        # Diferenças absolutas
        comparison['vehicles_diff'] = abs(metrics1.total_vehicles - metrics2.total_vehicles)
        comparison['distance_diff'] = abs(metrics1.total_distance - metrics2.total_distance)
        comparison['speed_diff'] = abs(metrics1.average_speed - metrics2.average_speed)
        comparison['travel_time_diff'] = abs(metrics1.average_travel_time - metrics2.average_travel_time)
        comparison['throughput_diff'] = abs(metrics1.throughput - metrics2.throughput)
        
        # Diferenças relativas (%)
        if metrics1.total_vehicles > 0:
            comparison['vehicles_rel_diff'] = comparison['vehicles_diff'] / metrics1.total_vehicles * 100
        
        if metrics1.average_speed > 0:
            comparison['speed_rel_diff'] = comparison['speed_diff'] / metrics1.average_speed * 100
        
        if metrics1.average_travel_time > 0:
            comparison['travel_time_rel_diff'] = comparison['travel_time_diff'] / metrics1.average_travel_time * 100
        
        return comparison
    
    def _events_to_dataframe(self, events: List[Any]) -> pd.DataFrame:
        """Converte lista de eventos para DataFrame"""
        
        if not events:
            return pd.DataFrame()
        
        data = []
        
        for event in events:
            row = {
                'timestamp': event.timestamp,
                'car_id': event.car_id,
                'event_type': event.event_type,
                'tick': event.tick
            }
            
            # Adicionar dados específicos baseado no tipo de evento
            if hasattr(event, 'data'):  # HTCEvent
                data_dict = event.data
                for key, value in data_dict.items():
                    if key not in row:
                        row[key] = value
            elif hasattr(event, 'attributes'):  # InterscsimulatorEvent
                attrs = event.attributes
                for key, value in attrs.items():
                    if key not in ['time', 'car_id', 'type', 'tick']:
                        try:
                            # Tentar converter para float se possível
                            row[key] = float(value)
                        except (ValueError, TypeError):
                            row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Garantir que colunas numéricas existam
        numeric_columns = [
            'cars_in_link', 'link_capacity', 'calculated_speed', 
            'free_speed', 'travel_time', 'total_distance', 'link_length'
        ]
        
        for col in numeric_columns:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df


# Exemplo de uso
if __name__ == "__main__":
    # Este seria usado com dados reais
    calculator = MetricsCalculator()
    print("Calculador de métricas inicializado")