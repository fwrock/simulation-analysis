"""
Modelos de dados para os simuladores
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


@dataclass
class VehicleEvent:
    """Evento base de veículo"""
    timestamp: float
    car_id: str
    event_type: str
    tick: int
    
    def __post_init__(self):
        self.timestamp = float(self.timestamp)
        self.tick = int(self.tick)


@dataclass
class HTCEvent(VehicleEvent):
    """Evento do simulador HTC"""
    simulation_id: str
    node_id: str
    report_type: str
    created_at: datetime
    data: Dict[str, Any]
    
    @classmethod
    def from_cassandra_row(cls, row):
        """Cria instância a partir de linha do Cassandra"""
        data = json.loads(row.data) if isinstance(row.data, str) else row.data
        
        return cls(
            timestamp=data.get('tick', 0),
            car_id=data.get('car_id', ''),
            event_type=data.get('event_type', ''),
            tick=data.get('tick', 0),
            simulation_id=row.simulation_id,
            node_id=row.node_id,
            report_type=row.report_type,
            created_at=row.created_at,
            data=data
        )


@dataclass
class JourneyStartedEvent(HTCEvent):
    """Evento de início de jornada"""
    origin: str
    destination: str
    route_cost: float
    route_length: int
    
    @classmethod
    def from_htc_event(cls, htc_event: HTCEvent):
        data = htc_event.data
        return cls(
            **htc_event.__dict__,
            origin=data.get('origin', ''),
            destination=data.get('destination', ''),
            route_cost=float(data.get('route_cost', 0)),
            route_length=int(data.get('route_length', 0))
        )


@dataclass
class EnterLinkEvent(HTCEvent):
    """Evento de entrada em link"""
    link_id: str
    link_length: float
    link_capacity: float
    cars_in_link: int
    free_speed: float
    calculated_speed: float
    travel_time: float
    lanes: int
    
    @classmethod
    def from_htc_event(cls, htc_event: HTCEvent):
        data = htc_event.data
        return cls(
            **htc_event.__dict__,
            link_id=data.get('link_id', ''),
            link_length=float(data.get('link_length', 0)),
            link_capacity=float(data.get('link_capacity', 0)),
            cars_in_link=int(data.get('cars_in_link', 0)),
            free_speed=float(data.get('free_speed', 0)),
            calculated_speed=float(data.get('calculated_speed', 0)),
            travel_time=float(data.get('travel_time', 0)),
            lanes=int(data.get('lanes', 0))
        )


@dataclass
class LeaveLinkEvent(HTCEvent):
    """Evento de saída de link"""
    link_id: str
    link_length: float
    total_distance: float
    
    @classmethod
    def from_htc_event(cls, htc_event: HTCEvent):
        data = htc_event.data
        return cls(
            **htc_event.__dict__,
            link_id=data.get('link_id', ''),
            link_length=float(data.get('link_length', 0)),
            total_distance=float(data.get('total_distance', 0))
        )


@dataclass
class JourneyCompletedEvent(HTCEvent):
    """Evento de jornada completada"""
    origin: str
    destination: str
    final_node: str
    reached_destination: bool
    total_distance: float
    best_cost: float
    
    @classmethod
    def from_htc_event(cls, htc_event: HTCEvent):
        data = htc_event.data
        return cls(
            **htc_event.__dict__,
            origin=data.get('origin', ''),
            destination=data.get('destination', ''),
            final_node=data.get('final_node', ''),
            reached_destination=bool(data.get('reached_destination', False)),
            total_distance=float(data.get('total_distance', 0)),
            best_cost=float(data.get('best_cost', 0))
        )


@dataclass
class InterscsimulatorEvent(VehicleEvent):
    """Evento do Interscsimulator"""
    attributes: Dict[str, Any]
    
    @classmethod
    def from_xml_element(cls, element):
        """Cria instância a partir de elemento XML"""
        attrs = dict(element.attrib)
        
        return cls(
            timestamp=float(attrs.get('time', 0)),
            car_id=attrs.get('car_id', ''),
            event_type=attrs.get('type', ''),
            tick=int(attrs.get('tick', attrs.get('time', 0))),
            attributes=attrs
        )


@dataclass
class SimulationSummary:
    """Resumo de uma simulação"""
    simulation_id: str
    simulator_type: str  # 'htc' ou 'interscsimulator'
    total_vehicles: int
    total_events: int
    simulation_duration: float
    start_time: float
    end_time: float
    events_by_type: Dict[str, int]
    
    def __post_init__(self):
        self.simulation_duration = self.end_time - self.start_time if self.end_time > self.start_time else 0


@dataclass
class ComparisonResult:
    """Resultado de comparação entre simulações"""
    htc_simulation_id: str
    interscsimulator_simulation_id: str
    similarity_score: float
    statistical_tests: Dict[str, Any]
    correlation_metrics: Dict[str, float]
    differences: Dict[str, Any]
    reproducibility_score: float